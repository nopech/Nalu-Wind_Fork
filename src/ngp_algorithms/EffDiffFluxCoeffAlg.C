/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "ngp_algorithms/EffDiffFluxCoeffAlg.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpTypes.h"
#include "Realm.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"

namespace sierra {
namespace nalu {

EffDiffFluxCoeffAlg::EffDiffFluxCoeffAlg(
  Realm& realm,
  stk::mesh::Part* part,
  ScalarFieldType* visc,
  ScalarFieldType* tvisc,
  ScalarFieldType* evisc,
  const double sigmaLam,
  const double sigmaTurb,
  const bool isTurbulent
) : Algorithm(realm, part),
    viscField_(visc),
    visc_(visc->mesh_meta_data_ordinal()),
    tvisc_(tvisc->mesh_meta_data_ordinal()),
    evisc_(evisc->mesh_meta_data_ordinal()),
    invSigmaLam_(1.0 / sigmaLam),
    invSigmaTurb_(1.0 / sigmaTurb),
    isTurbulent_(isTurbulent)
{}

void
EffDiffFluxCoeffAlg::execute()
{
  using Traits = nalu_ngp::NGPMeshTraits<ngp::Mesh>;

  const auto& meta = realm_.meta_data();

  stk::mesh::Selector sel = (
    meta.locally_owned_part() | meta.globally_shared_part())
    & stk::mesh::selectField(*viscField_);

  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto visc = fieldMgr.get_field<double>(visc_);
  const auto tvisc = fieldMgr.get_field<double>(tvisc_);
  auto evisc = fieldMgr.get_field<double>(evisc_);

  // Bring class variables into local scope for device capture
  const DblType invSigmaLam = invSigmaLam_;
  const DblType invSigmaTurb = invSigmaTurb_;

  if (isTurbulent_) {
    nalu_ngp::run_entity_algorithm(
      ngpMesh, stk::topology::NODE_RANK, sel,
      KOKKOS_LAMBDA(const Traits::MeshIndex& meshIdx) {
        evisc.get(meshIdx, 0) = (
          visc.get(meshIdx, 0) * invSigmaLam +
          tvisc.get(meshIdx, 0) * invSigmaTurb);
      });
  } else {
    nalu_ngp::run_entity_algorithm(
      ngpMesh, stk::topology::NODE_RANK, sel,
      KOKKOS_LAMBDA(const Traits::MeshIndex& meshIdx) {
        evisc.get(meshIdx, 0) = visc.get(meshIdx, 0) * invSigmaLam;
      });
  }

  // Set flag indicating that the field has been modified on device
  evisc.modify_on_device();
}

}  // nalu
}  // sierra
