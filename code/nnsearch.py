
# nnsearch.py
# nearest neighbor search

import faiss

def nnsearch(idxFeature, queryFeature, idxLabel, queryLabel):
    dimension = len(idxFeature[0])
    print('feature dimension = %d' % dimension)
    index = faiss.IndexFlatL2(dimension)
    res = faiss.StandardGpuResources()
    #res.setTempMemory(512 * 1024 * 1024)
    res.setTempMemory(512 * 2048 * 2048)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(np.ceil(idxFeature))
    print('total num index = %d' % gpu_index.ntotal)
    _, neighborMatrix = gpu_index.search(np.ceil(queryFeature), 100)
    #print(neighborMatrix.shape)

    label2result = {}
    for i in range(len(neighborMatrix)):
        neighbors = neighborMatrix[i]
        nbrLabels = ''
        for nbr in neighbors:
            nbrLabels += idxLabel[nbr]
            nbrLabels += ' '
        label2result[queryLabel[i]] = nbrLabels

    return label2result
