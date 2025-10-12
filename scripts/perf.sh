OMP_NUM_THREADS=1 perf stat -e instructions,cycles,stalled-cycles-frontend,stalled-cycles-backend,branches,branch-misses,dTLB-load-misses,LLC-store-misses,mem-loads,mem-stores \
  -r 1 ./build/project/examples/benchmark --run_test=performance_benchmark_all
