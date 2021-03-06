/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "ngp_algorithms/MdotEdgeAlg.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "Realm.h"
#include "utils/StkHelpers.h"

namespace sierra {
namespace nalu {

MdotEdgeAlg::MdotEdgeAlg(
  Realm& realm,
  stk::mesh::Part* part
) : Algorithm(realm, part),
    coordinates_(get_field_ordinal(realm.meta_data(), realm.get_coordinates_name())),
    velocityRTM_(get_field_ordinal(
      realm.meta_data(),
      realm.does_mesh_move() ? "velocity_rtm" : "velocity")),
    pressure_(get_field_ordinal(realm.meta_data(), "pressure")),
    densityNp1_(get_field_ordinal(realm.meta_data(), "density", stk::mesh::StateNP1)),
    Gpdx_(get_field_ordinal(realm.meta_data(), "dpdx")),
    edgeAreaVec_(get_field_ordinal(realm.meta_data(), "edge_area_vector", stk::topology::EDGE_RANK)),
    Udiag_(get_field_ordinal(realm.meta_data(), "momentum_diag")),
    massFlowRate_(
      get_field_ordinal(
        realm.meta_data(), "mass_flow_rate", stk::topology::EDGE_RANK))
{}

void
MdotEdgeAlg::execute()
{
  using EntityInfoType = nalu_ngp::EntityInfo<ngp::Mesh>;
  constexpr int NDimMax = 3;
  const auto& meta = realm_.meta_data();
  const int ndim = meta.spatial_dimension();

  const std::string dofName = "pressure";
  const DblType nocFac
    = (realm_.get_noc_usage(dofName)) ? 1.0 : 0.0;

  // Interpolation option for rho*U
  const DblType interpTogether = realm_.get_mdot_interp();
  const DblType om_interpTogether = (1.0 - interpTogether);

  // STK ngp::Field instances for capture by lambda
  const auto ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();
  const auto coordinates = fieldMgr.get_field<double>(coordinates_);
  const auto velocity = fieldMgr.get_field<double>(velocityRTM_);
  const auto Gpdx = fieldMgr.get_field<double>(Gpdx_);
  const auto density = fieldMgr.get_field<double>(densityNp1_);
  const auto pressure = fieldMgr.get_field<double>(pressure_);
  const auto udiag = fieldMgr.get_field<double>(Udiag_);
  const auto edgeAreaVec = fieldMgr.get_field<double>(edgeAreaVec_);
  auto mdot = fieldMgr.get_field<double>(massFlowRate_);

  const stk::mesh::Selector sel = meta.locally_owned_part()
    & stk::mesh::selectUnion(partVec_)
    & !(realm_.get_inactive_selector());

  nalu_ngp::run_edge_algorithm(
    ngpMesh, sel,
    KOKKOS_LAMBDA(const EntityInfoType& einfo) {
      NALU_ALIGNED DblType av[NDimMax];

      for (int d=0; d < ndim; ++d)
        av[d] = edgeAreaVec.get(einfo.meshIdx, d);

      const auto& nodes = einfo.entityNodes;
      const auto nodeL = ngpMesh.fast_mesh_index(nodes[0]);
      const auto nodeR = ngpMesh.fast_mesh_index(nodes[1]);

      const DblType pressureL = pressure.get(nodeL, 0);
      const DblType pressureR = pressure.get(nodeR, 0);

      const DblType densityL = density.get(nodeL, 0);
      const DblType densityR = density.get(nodeR, 0);

      const DblType udiagL = udiag.get(nodeL, 0);
      const DblType udiagR = udiag.get(nodeR, 0);

      const DblType projTimeScale = 0.5 * (1.0/udiagL + 1.0/udiagR);
      const DblType rhoIp = 0.5 * (densityL + densityR);

      DblType axdx = 0.0;
      DblType asq = 0.0;
      for (int d=0; d < ndim; ++d) {
        const DblType dxj = coordinates.get(nodeR, d) - coordinates.get(nodeL, d);
        asq += av[d] * av[d];
        axdx += av[d] * dxj;
      }
      const DblType inv_axdx = 1.0 / axdx;

      DblType tmdot = -projTimeScale * (pressureR - pressureL) * asq * inv_axdx;
      for (int d=0; d < ndim; ++d) {
        const DblType dxj =
          coordinates.get(nodeR, d) - coordinates.get(nodeL, d);
        // non-orthogonal correction
        const DblType kxj = av[d] - asq * inv_axdx * dxj;
        const DblType rhoUjIp = 0.5 * (densityR * velocity.get(nodeR, d) +
                                       densityL * velocity.get(nodeL, d));
        const DblType ujIp =
          0.5 * (velocity.get(nodeR, d) + velocity.get(nodeL, d));
        const DblType GjIp =
          0.5 * (Gpdx.get(nodeR, d) / udiagR + Gpdx.get(nodeL, d) / udiagL);
        tmdot += (interpTogether * rhoUjIp +
                  om_interpTogether * rhoIp * ujIp + GjIp) * av[d]
          - kxj * GjIp * nocFac;
      }

      // Update edge field
      mdot.get(einfo.meshIdx, 0) = tmdot;
    });

  // Flag that the field has been modified on device for future sync
  mdot.modify_on_device();
}


}  // nalu
}  // sierra
