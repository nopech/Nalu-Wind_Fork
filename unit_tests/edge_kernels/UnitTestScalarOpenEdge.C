/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "edge_kernels/ScalarEdgeOpenSolverAlg.h"

namespace {
namespace hex8_golds  {
static constexpr double rhs[8] = {
0, -20, -24.755282581476, 0, 0, -20, -22.795084971875, 0 
};

static constexpr double lhs[8][8] = {
{0, 0, 0, 0, 0, 0, 0, 0, },
{0, 10, 0, 0, 0, 0, 0, 0, },
{0, 0, 10, 0, 0, 0, 0, 0, },
{0, 0, 0, 0, 0, 0, 0, 0, },
{0, 0, 0, 0, 0, 0, 0, 0, },
{0, 0, 0, 0, 0, 10, 0, 0, },
{0, 0, 0, 0, 0, 0, 10, 0, },
{0, 0, 0, 0, 0, 0, 0, 0, },
};
}
}

TEST_F(SSTKernelHex8Mesh, NGP_scalar_edge_open_solver_alg)
{
  if (bulk_.parallel_size() > 1) return;

  const bool doPerturb = false;
  const bool generateSidesets = true;
  fill_mesh_and_init_fields(doPerturb, generateSidesets);

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;

  auto* part = meta_.get_part("surface_2");
  unit_test_utils::FaceElemHelperObjects helperObjs(
    bulk_, stk::topology::QUAD_4, stk::topology::HEX_8, 1, part);

  std::unique_ptr<sierra::nalu::Kernel> kernel(
    new sierra::nalu::ScalarEdgeOpenSolverAlg<
      sierra::nalu::AlgTraitsQuad4Hex8>(
      meta_, solnOpts_, tke_, tkebc_, dkdx_, tvisc_, 
      helperObjs.assembleFaceElemSolverAlg->faceDataNeeded_,
      helperObjs.assembleFaceElemSolverAlg->elemDataNeeded_));

  helperObjs.assembleFaceElemSolverAlg->activeKernels_.push_back(kernel.get());

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);

  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, hex8_golds::rhs, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<8>(
    helperObjs.linsys->lhs_, hex8_golds::lhs, 1.0e-12);
}
