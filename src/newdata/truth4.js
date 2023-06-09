            out.scalar = gcController.heapScan.Load()
        },
    },
    "/gc/scan/total:bytes": {
        compute: func(in *statAggregate, out *metricValue) {
            out.kind = metricKindUint64
            out.scalar = gcController.globalsScan.Load() + gcController.heapScan.Load() + gcController.lastStackScan.Load()
        },
    },
    "/gc/heap/allocs-by-size:bytes": {
        deps: makeStatDepSet(heapStatsDep),
        compute: func(in *statAggregate, out *metricValue) {
