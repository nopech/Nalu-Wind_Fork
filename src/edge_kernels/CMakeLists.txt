add_sources(GlobalSourceList
  # Edge kernels
  ContinuityEdgeSolverAlg.C
  MomentumEdgeSolverAlg.C
  ScalarEdgeSolverAlg.C
  WallDistEdgeSolverAlg.C

  # Face/edge BC kernels
  MomentumABLWallFuncEdgeKernel.C
  MomentumSymmetryEdgeKernel.C
  ScalarEdgeOpenSolverAlg.C
  )
