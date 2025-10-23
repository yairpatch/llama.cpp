#include "htp-dma.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#pragma clang diagnostic ignored "-Wunused-function"

static inline uint32_t pow2_ceil(uint32_t x) {
    if (x <= 1) {
        return 1;
    }
    int p = 2;
    x--;
    while (x >>= 1) {
        p <<= 1;
    }
    return p;
}

dma_queue * dma_queue_create(size_t capacity) {
    dma_queue * q = (dma_queue *) memalign(32, sizeof(dma_queue));
    if (q == NULL) {
        FARF(ERROR, "%s: failed to allocate DMA queue\n", __FUNCTION__);
        return NULL;
    }

    capacity = pow2_ceil(capacity);

    memset(q, 0, sizeof(dma_queue));
    q->capacity = capacity;
    q->idx_mask = capacity - 1;

    q->desc = (hexagon_udma_descriptor_type1_t *) memalign(64, capacity * sizeof(hexagon_udma_descriptor_type1_t));
    memset(q->desc, 0, capacity * sizeof(hexagon_udma_descriptor_type1_t));

    q->dst = (void **) memalign(4, capacity * sizeof(void *));
    memset(q->dst, 0, capacity * sizeof(void *));

    q->tail = &q->desc[capacity - 1];

    if (!q->desc && !q->dst) {
        FARF(ERROR, "%s: failed to allocate DMA queue items\n", __FUNCTION__);
        return NULL;
    }

    FARF(HIGH, "dma-queue: capacity %u\n", capacity);

    return q;
}

void dma_queue_delete(dma_queue * q) {
    if (!q) {
        return;
    }
    free(q->desc);
    free(q->dst);
    free(q);
}

void dma_queue_flush(dma_queue * q) {
    while (1) {
        uint32_t s = dmwait() & 0x3;
        if (s == HEXAGON_UDMA_DM0_STATUS_IDLE) {
            break;
        }
    }
    q->tail = NULL;
}
