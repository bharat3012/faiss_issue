from __future__ import print_function
from timedlogger import TimedLogger
import numpy as np
import time
import os
import sys
import faiss
import re

from multiprocessing.dummy import Pool as ThreadPool

dbname = None
index_key = None

# compute_new_index=False

INDEXPATH = "/tmp_host/"
ngpu = faiss.get_num_gpus()

replicas = 1  # nb of replicas of sharded dataset
add_batch_size = 32768
query_batch_size = 16384
# nprobes = [8] #[1 << l for l in range(9)]
knngraph = False
use_precomputed_tables = True
tempmem = -1  # if -1, use system default
max_add = -1
use_float16 = False
use_cache = True
nnn = 500
altadd = False
# I_fname = None
# D_fname = None

ncent = 100000


def generate_mmapped_data(outpath, n_samples, dim=128):
    T = TimedLogger("DB gen")
    np.random.seed(1234)  # make reproducible
    dtype = np.float32
    fp = np.memmap(outpath, dtype=dtype, mode="w+", shape=(n_samples, dim))
    n = 1000000
    i = 0
    while i < n_samples:
        print("Generating dataset", i // n, "/", n_samples // n)
        n_batch = n if i + n <= n_samples else n_samples - i
        xb = np.random.random(size=(n_batch, dim)).astype(dtype)
        # xb[:, 0] += (i + np.arange(n_batch)) / 1000.0

        fp[i : i + n_batch, :] = xb
        i += n_batch

    fp.flush()


def generate_data(size, dim=128):
    T = TimedLogger("DB gen")

    d = dim  # dimension
    nb = size  # database size
    # nq = 10000                       # nb of queries
    np.random.seed(1234)  # make reproducible
    xb = np.random.random((nb, d)).astype("float32")
    xb[:, 0] += np.arange(nb) / 1000.0
    # xq = np.random.random((nq, d)).astype('float32')
    # xq[:, 0] += np.arange(nq) / 1000.

    return xb


def dump_to_disk(xb, outpath):

    T = TimedLogger("disk write")
    dump_npy = "_" + str(xb.shape[0]) + ".npy"
    # dump_names = "_files_"+str(xb.shape[0])+".pickle"
    with open(outpath + dump_npy, "wb") as f:
        np.save(f, xb)

    return dump_npy


def load_from_disk(fname, shape):

    x = np.memmap(fname, dtype="float32", mode="r", shape=shape)
    return x.view("float32")


def rate_limited_imap(f, l):
    """A threaded imap that does not produce elements faster than they
    are consumed"""
    pool = ThreadPool(1)
    res = None
    for i in l:
        res_next = pool.apply_async(f, (i,))
        if res:
            yield res.get()
        res = res_next
    yield res.get()


class IdentPreproc:
    """a pre-processor is either a faiss.VectorTransform or an IndentPreproc"""

    def __init__(self, d):
        self.d_in = self.d_out = d

    def apply_py(self, x):
        return x


def sanitize(x):
    """convert array to a c-contiguous float array"""
    return np.ascontiguousarray(x.astype("float32"))


def dataset_iterator(x, preproc, bs):
    """iterate over the lines of x in blocks of size bs"""

    nb = x.shape[0]
    block_ranges = [(i0, min(nb, i0 + bs)) for i0 in range(0, nb, bs)]

    def prepare_block(i01):
        i0, i1 = i01
        xb = sanitize(x[i0:i1])
        return i0, preproc.apply_py(xb)

    return rate_limited_imap(prepare_block, block_ranges)


#################################################################
# Wake up GPUs
#################################################################

print("preparing resources for %d GPUs" % ngpu)

gpu_resources = []

for i in range(ngpu):
    res = faiss.StandardGpuResources()
    if tempmem >= 0:
        res.setTempMemory(tempmem)
    gpu_resources.append(res)


def make_vres_vdev(i0=0, i1=-1):

    T = TimedLogger("make_vres_vdev")
    " return vectors of device ids and resources useful for gpu_multiple"
    vres = faiss.GpuResourcesVector()
    vdev = faiss.IntVector()
    if i1 == -1:
        i1 = ngpu
    for i in range(i0, i1):
        vdev.push_back(i)
        vres.push_back(gpu_resources[i])
    return vres, vdev


def train_coarse_quantizer(x, k, preproc):

    T = TimedLogger("train_coarse_quantizer")

    d = preproc.d_out
    clus = faiss.Clustering(d, k)
    clus.niter = 18
    clus.verbose = True
    # clus.niter = 2
    clus.max_points_per_centroid = 10000000

    print("apply preproc on shape", x.shape, "k=", k)
    t0 = time.time()
    x = preproc.apply_py(sanitize(x))
    print("   preproc %.3f s output shape %s" % (time.time() - t0, x.shape))

    vres, vdev = make_vres_vdev()
    # index = faiss.index_cpu_to_gpu_multiple(
    #    vres, vdev, faiss.IndexFlatL2(d))

    index = faiss.index_cpu_to_gpu_multiple(vres, vdev, faiss.IndexFlatIP(d))
    clus.train(x, index)
    centroids = faiss.vector_float_to_array(clus.centroids)

    return centroids.reshape(k, d)


def prepare_coarse_quantizer(preproc, xt):

    T = TimedLogger("prepare_coarse_quantizer")
    # ncent = 100000
    global ncent
    # ncent = min(100000,int(xt.shape[0]/30))
    ncent = 16384  # 16384  # int(xt.shape[0] / 40)

    if True:
        nt = max(1000000, 256 * ncent, xt.shape[0])
        print("train coarse quantizer...")
        t0 = time.time()
        centroids = train_coarse_quantizer(xt[:nt], ncent, preproc)
        print("Coarse train time: %.3f s" % (time.time() - t0))
        if False:
            print("store centroids", cent_cachefile)
            np.save(cent_cachefile, centroids)

    coarse_quantizer = faiss.IndexFlatL2(preproc.d_out)
    coarse_quantizer.add(centroids)

    return coarse_quantizer


def prepare_trained_index(preproc, xt):

    T = TimedLogger("prepare_trained_index")

    coarse_quantizer = prepare_coarse_quantizer(preproc, xt)
    d = preproc.d_out
    if True:  # pqflat_str == 'Flat':
        print("making an IVFFlat index")
        # idx_model = faiss.IndexIVFFlat(coarse_quantizer, d, ncent,
        #                               faiss.METRIC_L2)
        idx_model = faiss.IndexIVFFlat(
            coarse_quantizer, d, ncent, faiss.METRIC_INNER_PRODUCT
        )
    else:
        m = int(pqflat_str[2:])
        assert m < 56 or use_float16, "PQ%d will work only with -float16" % m
        print("making an IVFPQ index, m = ", m)
        idx_model = faiss.IndexIVFPQ(coarse_quantizer, d, ncent, m, 8)

    coarse_quantizer.this.disown()
    idx_model.own_fields = True

    # finish training on CPU
    t0 = time.time()
    print("Training vector codes")
    x = preproc.apply_py(sanitize(xt))  # [:1000000]))
    idx_model.train(x)
    print("  done %.3f s" % (time.time() - t0))

    return idx_model


def get_preprocessor(xb, dout, pca=False):

    T = TimedLogger("get_preprocessor")
    preproc = None
    dout = dout
    if pca:
        preproc = faiss.PCAMatrix(xb.shape[1], dout, 0, True)
        preproc.train(sanitize(xb))

    else:  # False:
        d = xb.shape[1]
        preproc = IdentPreproc(d)
    return preproc


def compute_populated_index(preproc, xb):
    """Add elements to a sharded index. Return the index and if available
    a sharded gpu_index that contains the same data."""

    indexall = prepare_trained_index(preproc, xb)

    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = use_float16
    co.useFloat16CoarseQuantizer = False
    co.usePrecomputed = use_precomputed_tables
    co.indicesOptions = faiss.INDICES_CPU
    co.verbose = True
    co.reserveVecs = max_add if max_add > 0 else xb.shape[0]
    co.shard = True
    assert co.shard_type in (0, 1, 2)
    vres, vdev = make_vres_vdev()
    gpu_index = faiss.index_cpu_to_gpu_multiple(vres, vdev, indexall, co)

    print("add...")
    t0 = time.time()
    nb = xb.shape[0]
    for i0, xs in dataset_iterator(xb, preproc, add_batch_size):
        i1 = i0 + xs.shape[0]
        gpu_index.add_with_ids(xs, np.arange(i0, i1))
        if max_add > 0 and gpu_index.ntotal > max_add:
            print("Flush indexes to CPU")
            for i in range(ngpu):
                index_src_gpu = faiss.downcast_index(gpu_index.at(i))
                index_src = faiss.index_gpu_to_cpu(index_src_gpu)
                print("  index %d size %d" % (i, index_src.ntotal))
                index_src.copy_subset_to(indexall, 0, 0, nb)
                index_src_gpu.reset()
                index_src_gpu.reserveMemory(max_add)
            gpu_index.sync_with_shard_indexes()

        print("\r%d/%d (%.3f s)  " % (i0, nb, time.time() - t0), end=" ")
        sys.stdout.flush()
    print("Add time: %.3f s" % (time.time() - t0))

    print("Aggregate indexes to CPU")
    t0 = time.time()

    if hasattr(gpu_index, "at"):
        # it is a sharded index
        for i in range(ngpu):
            index_src = faiss.index_gpu_to_cpu(gpu_index.at(i))
            print("  index %d size %d" % (i, index_src.ntotal))
            index_src.copy_subset_to(indexall, 0, 0, nb)
    else:
        # simple index
        index_src = faiss.index_gpu_to_cpu(gpu_index)
        index_src.copy_subset_to(indexall, 0, 0, nb)

    print("  done in %.3f s" % (time.time() - t0))

    if max_add > 0:
        # it does not contain all the vectors
        gpu_index = None

    return gpu_index, indexall


def compute_populated_index_2(preproc, xb):

    T = TimedLogger("compute_populated_index")

    indexall = prepare_trained_index(preproc, xb)

    # set up a 3-stage pipeline that does:
    # - stage 1: load + preproc
    # - stage 2: assign on GPU
    # - stage 3: add to index

    stage1 = dataset_iterator(xb, preproc, add_batch_size)

    vres, vdev = make_vres_vdev()
    coarse_quantizer_gpu = faiss.index_cpu_to_gpu_multiple(
        vres, vdev, indexall.quantizer
    )

    def quantize(args):
        (i0, xs) = args
        _, assign = coarse_quantizer_gpu.search(xs, 1)
        return i0, xs, assign.ravel()

    stage2 = rate_limited_imap(quantize, stage1)

    print("add...")
    t0 = time.time()
    nb = xb.shape[0]

    for i0, xs, assign in stage2:
        i1 = i0 + xs.shape[0]
        if indexall.__class__ == faiss.IndexIVFPQ:
            indexall.add_core_o(
                i1 - i0, faiss.swig_ptr(xs), None, None, faiss.swig_ptr(assign)
            )
        elif indexall.__class__ == faiss.IndexIVFFlat:
            indexall.add_core(i1 - i0, faiss.swig_ptr(xs), None, faiss.swig_ptr(assign))
        else:
            assert False

        print("\r%d/%d (%.3f s)  " % (i0, nb, time.time() - t0), end=" ")
        sys.stdout.flush()
    print("Add time: %.3f s" % (time.time() - t0))

    return None, indexall


def get_populated_index(preproc, xt, compute_new_index=True, index_file=None):

    index_cachefile = INDEXPATH + index_file if index_file else None
    # faiss.normalize_L2(xt) ##normalize for IP

    T = TimedLogger("get_populated_index")

    if compute_new_index:
        gpu_index, indexall = compute_populated_index_2(preproc, xt)
        # gpu_index, indexall = compute_populated_index(preproc,xt)

        if index_cachefile:
            print("store", index_cachefile)
            faiss.write_index(indexall, index_cachefile)
    else:
        print("load", index_cachefile)
        indexall = faiss.read_index(index_cachefile)
        gpu_index = None

    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = use_float16
    co.useFloat16CoarseQuantizer = False
    co.usePrecomputed = use_precomputed_tables
    co.indicesOptions = 0
    co.verbose = True
    co.shard = True  # the replicas will be made "manually"
    t0 = time.time()
    print("CPU index contains %d vectors, move to GPU" % indexall.ntotal)
    if replicas == 1:

        if not gpu_index:
            print("copying loaded index to GPUs")
            vres, vdev = make_vres_vdev()
            index = faiss.index_cpu_to_gpu_multiple(vres, vdev, indexall, co)
        else:
            index = gpu_index

    else:
        del gpu_index  # We override the GPU index

        print("Copy CPU index to %d sharded GPU indexes" % replicas)

        index = faiss.IndexReplicas()

        for i in range(replicas):
            gpu0 = ngpu * i / replicas
            gpu1 = ngpu * (i + 1) / replicas
            vres, vdev = make_vres_vdev(gpu0, gpu1)

            print("   dispatch to GPUs %d:%d" % (gpu0, gpu1))

            index1 = faiss.index_cpu_to_gpu_multiple(vres, vdev, indexall, co)
            index1.this.disown()
            index.addIndex(index1)
        index.own_fields = True
    del indexall
    print("move to GPU done in %.3f s" % (time.time() - t0))
    return index


def eval_dataset(index, preproc, xq, nprobes, I_fname, D_fname):

    T = TimedLogger("Eval_DB")

    # faiss.normalize_L2(xq)
    ps = faiss.GpuParameterSpace()
    ps.initialize(index)
    print(f"index with d = {index.d}")
    gt_I = xq
    nq_gt = xq.shape[0]  # gt_I.shape[0]

    print(f"search...{xq.shape[1]}d vectors")
    sl = query_batch_size
    nq = xq.shape[0]
    for nprobe in nprobes:
        print("nprobe : ", nprobe)
        if nprobe > ncent:
            break
        ps.set_index_parameter(index, "nprobe", nprobe)
        t0 = time.time()

        if sl == 0:
            D, I = index.search(preproc.apply_py(sanitize(xq)), nnn)
        else:
            I = np.empty((nq, nnn), dtype="int32")
            D = np.empty((nq, nnn), dtype="float32")

            inter_res = ""

            for i0, xs in dataset_iterator(xq, preproc, sl):
                print(
                    "\r%d/%d (%.3f s%s)   " % (i0, nq, time.time() - t0, inter_res),
                    end=" ",
                )
                sys.stdout.flush()

                i1 = i0 + xs.shape[0]
                Di, Ii = index.search(xs, nnn)

                I[i0:i1] = Ii
                D[i0:i1] = Di

                if knngraph and not inter_res and i1 >= nq_gt:
                    ires = eval_intersection_measure(gt_I[:, :nnn], I[:nq_gt])

        t1 = time.time()
        if False:
            ires = eval_intersection_measure(gt_I[:, :nnn], I[:nq_gt])
            print(
                "  probe=%-3d: %.3f s rank-%d intersection results: %.4f"
                % (nprobe, t1 - t0, nnn, ires)
            )
        elif False:
            print("  probe=%-3d: %.3f s" % (nprobe, t1 - t0), end=" ")
            gtc = gt_I[:, :1]
            nq = xq.shape[0]
            for rank in 1, 10, 100:
                if rank > nnn:
                    continue
                nok = (I[:, :rank] == gtc).sum()
                print("1-R@%d: %.4f" % (rank, nok / float(nq)), end=" ")
            print()
        if I_fname:
            I_fname_i = I_fname + "_" + str(I.shape[0]) + "_" + str(nprobe)
            print("storing", I_fname_i)
            with open(I_fname_i, "wb") as f:
                np.save(f, I)
        if D_fname:
            D_fname_i = D_fname + "_" + str(I.shape[0]) + "_" + str(nprobe)
            print("storing", D_fname_i)
            with open(D_fname_i, "wb") as f:
                np.save(f, D)
