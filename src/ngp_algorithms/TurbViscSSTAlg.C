/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "ngp_algorithms/TurbViscSSTAlg.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpTypes.h"
#include "Realm.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"

namespace sierra{
namespace nalu{

TurbViscSSTAlg::TurbViscSSTAlg(
  Realm &realm,
  stk::mesh::Part *part,
  ScalarFieldType* tvisc
) : Algorithm(realm, part),
    tviscField_(tvisc),
    density_(get_field_ordinal(realm.meta_data(), "density")),
    viscosity_(get_field_ordinal(realm.meta_data(), "viscosity")),
    tke_(get_field_ordinal(realm.meta_data(), "turbulent_ke")),
    sdr_(get_field_ordinal(realm.meta_data(), "specific_dissipation_rate")),
    minDistance_(get_field_ordinal(realm.meta_data(), "minimum_distance_to_wall")),
    dudx_(get_field_ordinal(realm.meta_data(), "dudx")),
    tvisc_(tvisc->mesh_meta_data_ordinal()),
    aOne_(realm.get_turb_model_constant(TM_aOne)),
    betaStar_(realm.get_turb_model_constant(TM_betaStar))
{}

void
TurbViscSSTAlg::execute()
{
  using Traits = nalu_ngp::NGPMeshTraits<ngp::Mesh>;

  const auto& meta = realm_.meta_data();

  stk::mesh::Selector sel = (
    meta.locally_owned_part() | meta.globally_shared_part())
    & stk::mesh::selectField(*tviscField_);

  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto density = fieldMgr.get_field<double>(density_);
  const auto visc = fieldMgr.get_field<double>(viscosity_);
  const auto tke = fieldMgr.get_field<double>(tke_);
  const auto sdr = fieldMgr.get_field<double>(sdr_);
  const auto minD = fieldMgr.get_field<double>(minDistance_);
  const auto dudx = fieldMgr.get_field<double>(dudx_);
  auto tvisc = fieldMgr.get_field<double>(tvisc_);

  const DblType aOne = aOne_;
  const DblType betaStar = betaStar_;
  const int nDim = meta.spatial_dimension();

  nalu_ngp::run_entity_algorithm(
    ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const Traits::MeshIndex& meshIdx) {
      DblType sijMag = 0.0;
      for ( int i = 0; i < nDim; ++i ) {
        const int offSet = nDim*i;
        for ( int j = 0; j < nDim; ++j ) {
          const DblType rateOfStrain = 0.5*(dudx.get(meshIdx, offSet+j) + dudx.get(meshIdx, nDim*j+i));
          sijMag += rateOfStrain*rateOfStrain;
        }
      }
      sijMag = std::sqrt(2.0*sijMag);

      const DblType minDSq = minD.get(meshIdx, 0)*minD.get(meshIdx, 0);
      const DblType trbDiss = std::sqrt(tke.get(meshIdx, 0))/betaStar/sdr.get(meshIdx, 0)/minD.get(meshIdx, 0);
      const DblType lamDiss = 500.0*visc.get(meshIdx, 0)/density.get(meshIdx, 0)/sdr.get(meshIdx, 0)/minDSq;
      const DblType fArgTwo = std::max(2.0*trbDiss, lamDiss);
      const DblType fTwo = std::tanh(fArgTwo*fArgTwo);

      tvisc.get(meshIdx, 0) = aOne*density.get(meshIdx, 0)*tke.get(meshIdx, 0)/std::max(aOne*sdr.get(meshIdx, 0), sijMag*fTwo);
    });
}

} // namespace nalu
} // namespace Sierra
