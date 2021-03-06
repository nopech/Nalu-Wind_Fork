/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "ngp_algorithms/UnitTestNgpAlgUtils.h"
#include "ngp_utils/NgpLoopUtils.h"

namespace unit_test_alg_utils
{

void
linear_scalar_field(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  ScalarFieldType& field,
  const double xCoeff,
  const double yCoeff,
  const double zCoeff)
{
  using Traits = sierra::nalu::nalu_ngp::NGPMeshTraits<ngp::Mesh>;
  ngp::Mesh ngpMesh(bulk);
  ngp::Field<double> coords(bulk, coordinates);
  ngp::Field<double> ngpField(bulk, field);

  const stk::mesh::Selector sel = bulk.mesh_meta_data().universal_part();

  sierra::nalu::nalu_ngp::run_entity_algorithm(
    ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const typename Traits::MeshIndex& meshIdx) {
      ngpField.get(meshIdx, 0) =
        coords.get(meshIdx, 0) * xCoeff +
        coords.get(meshIdx, 1) * yCoeff +
        coords.get(meshIdx, 2) * zCoeff;
    });

  ngpField.modify_on_device();
  ngpField.sync_to_host();
}

void
linear_scalar_field(
  const stk::mesh::BulkData& bulk,
  const VectorFieldType& coordinates,
  VectorFieldType& field,
  const double xCoeff,
  const double yCoeff,
  const double zCoeff)
{
  using Traits = sierra::nalu::nalu_ngp::NGPMeshTraits<ngp::Mesh>;
  ngp::Mesh ngpMesh(bulk);
  ngp::Field<double> coords(bulk, coordinates);
  ngp::Field<double> ngpField(bulk, field);

  const stk::mesh::Selector sel = bulk.mesh_meta_data().universal_part();

  sierra::nalu::nalu_ngp::run_entity_algorithm(
    ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const typename Traits::MeshIndex& meshIdx) {
      ngpField.get(meshIdx, 0) = coords.get(meshIdx, 0) * xCoeff;
      ngpField.get(meshIdx, 1) = coords.get(meshIdx, 1) * yCoeff;
      ngpField.get(meshIdx, 2) = coords.get(meshIdx, 2) * zCoeff;
    });

  ngpField.modify_on_device();
  ngpField.sync_to_host();
}

} // namespace
