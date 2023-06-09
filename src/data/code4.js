            out.scalar = gcController.globalsScan.Load()
        },
    },
    "/gc/scan/heap:bytes": {
        compute: func(in *statAggregate, out *metricValue) {
            out.kind = metricKindUint64
            out.scalar = gcController.heapScan.Load()
        },
    },
    "/gc/heap/allocs-by-size:bytes": {
        deps: makeStatDepSet(heapStatsDep),
        compute: func(in *statAggregate, out *metricValue) {
