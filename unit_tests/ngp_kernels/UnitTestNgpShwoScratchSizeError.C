/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "gtest/gtest.h"
#include "UnitTestUtils.h"
#include <NaluEnv.h>

#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>

#include "ngp_utils/NgpTypes.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include <ngp_utils/NgpMEUtils.h>

#include "ElemDataRequests.h"
#include "ElemDataRequestsGPU.h"

#include <master_element/MasterElement.h>
#include <master_element/MasterElementFunctions.h>
#include <master_element/MasterElementUtils.h>
#include <master_element/TensorOps.h>
#include <master_element/Hex8GeometryFunctions.h>

//#include <FORTRAN_Proto.h>
#include <KokkosInterface.h>
#include <SimdInterface.h>
#include <MultiDimViews.h>

#include <iostream>
#include <cmath>
#include <array>
#include <set>
#include <type_traits>

//--------------------------------------------------------------------------

namespace sierra{
namespace nalu{
namespace Impl {

template<typename T, typename SHMEM>
class MasterElementViews
{
public:
  KOKKOS_FUNCTION MasterElementViews() = default;
  KOKKOS_FUNCTION virtual ~MasterElementViews() = default;
  KOKKOS_FUNCTION void fill_master_element_views_new_me( MasterElement* ); 

  SharedMemView<T**, SHMEM> fc_areav;
  SharedMemView<T**, SHMEM> scs_areav;
  SharedMemView<T***, SHMEM> dndx_fc_scs;
  SharedMemView<T***, SHMEM> dndx_shifted_fc_scs;
  SharedMemView<T***, SHMEM> dndx;
  SharedMemView<T***, SHMEM> dndx_shifted;
  SharedMemView<T***, SHMEM> dndx_scv;
  SharedMemView<T***, SHMEM> dndx_scv_shifted;
  SharedMemView<T***, SHMEM> dndx_fem;
  SharedMemView<T***, SHMEM> deriv_fc_scs;
  SharedMemView<T***, SHMEM> deriv;
  SharedMemView<T***, SHMEM> deriv_scv;
  SharedMemView<T***, SHMEM> deriv_fem;
  SharedMemView<T*, SHMEM> det_j_fc_scs;
  SharedMemView<T*, SHMEM> det_j;
  SharedMemView<T*, SHMEM> det_j_scv;
  int big_ol_array[100];
};

template<typename T, typename SHMEM=HostShmem>
class ScratchViews
{
public:
  KOKKOS_FUNCTION ScratchViews() = default;
  KOKKOS_FUNCTION ~ScratchViews() = default;
  KOKKOS_INLINE_FUNCTION MasterElementViews<T, SHMEM>& get_me_views(const COORDS_TYPES cType)
  {
    return meViews[cType];
  }

private:
  MasterElementViews<T, SHMEM> meViews[MAX_COORDS_TYPES];
};

template<typename T, typename SHMEM>
void MasterElementViews<T, SHMEM>::fill_master_element_views_new_me( MasterElement* meSCS)
{
  SharedMemView<DoubleType**, SHMEM> coordsView;
  meSCS->face_grad_op(0, coordsView, dndx_fc_scs);
}

} //  namespace Impl



namespace UnitTestHex8 {

template<typename ViewType>
KOKKOS_INLINE_FUNCTION
void hex8_shape_fcn(
  const int      npts,
  const double * isoParCoord,
  ViewType &shape_fcn)
{
  const DoubleType half   = 0.50;
  const DoubleType one4th = 0.25;
  const DoubleType one8th = 0.125;
  for ( int j = 0; j < npts; ++j ) {

    const DoubleType s1 = isoParCoord[j*3];
    const DoubleType s2 = isoParCoord[j*3+1];
    const DoubleType s3 = isoParCoord[j*3+2];

    shape_fcn(j,0) = one8th + one4th*(-s1 - s2 - s3) + half*( s2*s3 + s3*s1 + s1*s2 ) - s1*s2*s3;
    shape_fcn(j,1) = one8th + one4th*( s1 - s2 - s3) + half*( s2*s3 - s3*s1 - s1*s2 ) + s1*s2*s3;
    shape_fcn(j,2) = one8th + one4th*( s1 + s2 - s3) + half*(-s2*s3 - s3*s1 + s1*s2 ) - s1*s2*s3;
    shape_fcn(j,3) = one8th + one4th*(-s1 + s2 - s3) + half*(-s2*s3 + s3*s1 - s1*s2 ) + s1*s2*s3;
    shape_fcn(j,4) = one8th + one4th*(-s1 - s2 + s3) + half*(-s2*s3 - s3*s1 + s1*s2 ) + s1*s2*s3;
    shape_fcn(j,5) = one8th + one4th*( s1 - s2 + s3) + half*(-s2*s3 + s3*s1 - s1*s2 ) - s1*s2*s3;
    shape_fcn(j,6) = one8th + one4th*( s1 + s2 + s3) + half*( s2*s3 + s3*s1 + s1*s2 ) + s1*s2*s3;
    shape_fcn(j,7) = one8th + one4th*(-s1 + s2 + s3) + half*( s2*s3 - s3*s1 - s1*s2 ) - s1*s2*s3;
  }
}

// Hex 8 subcontrol volume
class HexSCV : public MasterElement
{
public:
  using AlgTraits = AlgTraitsHex8;

  KOKKOS_FUNCTION
  HexSCV();

  KOKKOS_FUNCTION
  virtual ~HexSCV() = default;

  using MasterElement::determinant;
  using MasterElement::shifted_grad_op;
  using MasterElement::shifted_shape_fcn;
  using MasterElement::shape_fcn;

  // NGP-ready methods first
  KOKKOS_FUNCTION void shape_fcn(SharedMemView<DoubleType**, DeviceShmem> &shpfc);

  KOKKOS_FUNCTION void shifted_shape_fcn(
    SharedMemView<DoubleType**, DeviceShmem> &shpfc);

  KOKKOS_FUNCTION void determinant(
    SharedMemView<DoubleType**, DeviceShmem>& coords,
    SharedMemView<DoubleType*, DeviceShmem>& volume);

  KOKKOS_FUNCTION void grad_op(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType***, DeviceShmem>&,
    SharedMemView<DoubleType***, DeviceShmem>&);

  KOKKOS_FUNCTION void shifted_grad_op(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType***, DeviceShmem>&,
    SharedMemView<DoubleType***, DeviceShmem>&);

  KOKKOS_FUNCTION void Mij(
    SharedMemView<DoubleType**, DeviceShmem>& ,
    SharedMemView<DoubleType***, DeviceShmem>& ,
    SharedMemView<DoubleType***, DeviceShmem>& );

  void determinant( const int, const double*, double*, double*);
  void grad_op( const int, const double*, double*, double*, double*, double*);
  void Mij( const double *, double *, double *);
  template<typename ViewType> KOKKOS_FUNCTION void shape_fcn(ViewType &);
  void shape_fcn( double *);
  void shifted_shape_fcn( double *);

  static constexpr int nDim_ = AlgTraits::nDim_;
  static constexpr int nodesPerElement_ = AlgTraits::nodesPerElement_;
  static constexpr int numIntPoints_ = AlgTraits::numScvIp_;
 
   // standard integration location
  const double intgLoc_[numIntPoints_*nDim_] = {
   -0.25,  -0.25,  -0.25,
   +0.25,  -0.25,  -0.25,
   +0.25,  +0.25,  -0.25,
   -0.25,  +0.25,  -0.25,
   -0.25,  -0.25,  +0.25,
   +0.25,  -0.25,  +0.25,
   +0.25,  +0.25,  +0.25,
   -0.25,  +0.25,  +0.25};
 
  // shifted integration location
  const double intgLocShift_[24] = {
   -0.5,  -0.5,  -0.5,
   +0.5,  -0.5,  -0.5,
   +0.5,  +0.5,  -0.5,
   -0.5,  +0.5,  -0.5,
   -0.5,  -0.5,  +0.5,
   +0.5,  -0.5,  +0.5,
   +0.5,  +0.5,  +0.5,
   -0.5,  +0.5,  +0.5};

};

template<typename ViewType> KOKKOS_FUNCTION void
HexSCV::shape_fcn(ViewType &shpfc)
{ hex8_shape_fcn(numIntPoints_, &intgLoc_[0], shpfc); }

HexSCV::HexSCV() : MasterElement()
{
  MasterElement::nDim_                  = nDim_;
  MasterElement::nodesPerElement_       = nodesPerElement_;
  MasterElement::numIntPoints_          = numIntPoints_;
}

void HexSCV::determinant( const int, const double*, double*, double*) {}

void HexSCV::shape_fcn(SharedMemView<DoubleType**, DeviceShmem> &shpfc)
{ hex8_shape_fcn(numIntPoints_, &intgLoc_[0], shpfc); }

void HexSCV::shifted_shape_fcn(SharedMemView<DoubleType**, DeviceShmem> &shpfc)
{ hex8_shape_fcn(numIntPoints_, &intgLocShift_[0], shpfc); }

void HexSCV::determinant(
  SharedMemView<DoubleType**, DeviceShmem>& ,
  SharedMemView<DoubleType*, DeviceShmem>& ) { }
void HexSCV::grad_op(
  SharedMemView<DoubleType**, DeviceShmem>&,
  SharedMemView<DoubleType***, DeviceShmem>&,
  SharedMemView<DoubleType***, DeviceShmem>&) { }
void HexSCV::grad_op( const int, const double*, double*, double*, double*, double*) {}
void HexSCV::shifted_grad_op(
  SharedMemView<DoubleType**, DeviceShmem>&,
  SharedMemView<DoubleType***, DeviceShmem>&,
  SharedMemView<DoubleType***, DeviceShmem>&) { }
void HexSCV::shifted_shape_fcn(double *){}
void HexSCV::shape_fcn(double *) { }
void HexSCV::Mij( const double *, double *, double *) { }
void HexSCV::Mij(
    SharedMemView<DoubleType**, DeviceShmem>& ,
    SharedMemView<DoubleType***, DeviceShmem>& ,
    SharedMemView<DoubleType***, DeviceShmem>& ) { }
}


namespace nalu_ngp {

class NgpViewTest : public ::testing::Test
{
public:
  NgpViewTest()
    : meta(3),
      bulk(meta, MPI_COMM_WORLD)
  {}

  ~NgpViewTest() = default;

  void fill_mesh_and_init_fields(const std::string& meshSpec = "generated:2x2x2")
  {
    unit_test_utils::fill_hex8_mesh(meshSpec, bulk);
    coordField = static_cast<const VectorFieldType*>(meta.coordinate_field());
    EXPECT_TRUE(coordField != nullptr);
  }

  stk::mesh::MetaData meta;
  stk::mesh::BulkData bulk;
  const VectorFieldType* coordField{nullptr};
};

void fill_master_element_view_test(stk::mesh::BulkData &bulk)
{
  const auto& meta = bulk.mesh_meta_data();
  sierra::nalu::ElemDataRequests dataReq(meta);
  auto meSCV = sierra::nalu::create_device_expression<sierra::nalu::UnitTestHex8::HexSCV>();
 
  sierra::nalu::nalu_ngp::MeshInfo<> meshInfo(bulk);
  stk::mesh::Selector sel = meta.universal_part();

  using Traits         = NGPMeshTraits<ngp::Mesh>;
  using TeamPolicy     = typename Traits::TeamPolicy;
  using TeamHandleType = typename Traits::TeamHandleType;
  using ShmemType      = typename Traits::ShmemType;

  const stk::topology::rank_t rank = stk::topology::ELEM_RANK;
  const sierra::nalu::ElemDataRequests &dataReqs = dataReq;

  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  sierra::nalu::ElemDataRequestsGPU dataReqNGP(fieldMgr, dataReqs, meshInfo.num_fields());

  const int bytes_per_team = 0;
  const int bytes_per_thread = 0;
  const auto& buckets = ngpMesh.get_bucket_ids(rank, sel);
  auto team_exec = impl::ngp_mesh_team_policy<TeamPolicy>(
    buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(team_exec, KOKKOS_LAMBDA(const TeamHandleType& team) {
    DoubleType unmanaged[64];
    sierra::nalu::SharedMemView<DoubleType**, DeviceShmem> scv_shifted_shape_fcn(unmanaged,8,8);
    meSCV->shifted_shape_fcn(scv_shifted_shape_fcn);

    Impl::ScratchViews<DoubleType, ShmemType> prereqData;
    auto bktId = buckets.device_get(team.league_rank());
    const size_t bktLen = ngpMesh.get_bucket(rank, bktId).size();
    const size_t simdBktLen = get_num_simd_groups(bktLen);
    Kokkos::parallel_for(
    Kokkos::TeamThreadRange(team, simdBktLen), [&](const size_t& ) {
      MasterElement *meSCS = dataReqNGP.get_cvfem_surface_me();
      const typename sierra::nalu::ElemDataRequestsGPU::CoordsTypesView& coordsTypes = dataReqNGP.get_coordinates_types();
      for(unsigned i=0; i<coordsTypes.size(); ++i) {
        auto cType = coordsTypes(0);
        auto& meData = prereqData.get_me_views(cType);
        meData.fill_master_element_views_new_me(meSCS);
      }
    });
  });
}

TEST_F(NgpViewTest, DISABLE_NGP_show_scratch_size_error)
{
  fill_mesh_and_init_fields("generated:2x2x2");
  fill_master_element_view_test(bulk);
}

TEST_F(NgpViewTest, NGP_show_scratch_size_no_error)
{
#ifdef KOKKOS_ENABLE_CUDA
  // FYI: Nvidia default: 1024
  //      KOKKOS default: 2048
  //      Suggested Nalu: 4096
  cudaDeviceSetLimit (cudaLimitStackSize, 4096);
#endif
  fill_mesh_and_init_fields("generated:2x2x2");
  EXPECT_NO_THROW(fill_master_element_view_test(bulk));
}
}}}
