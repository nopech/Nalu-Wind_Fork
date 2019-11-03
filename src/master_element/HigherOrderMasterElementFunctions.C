/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <master_element/HigherOrderMasterElementFunctions.h>

#include <master_element/MasterElement.h>

#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/LagrangeBasis.h>
#include <element_promotion/NodeMapMaker.h>



#include <AlgTraits.h>
#include <KokkosInterface.h>

#include <array>

namespace sierra {
namespace nalu {

void gradient_2d(
    int nodesPerElement,
    const double* POINTER_RESTRICT elemNodalCoords,
    const double* POINTER_RESTRICT shapeDeriv,
    double* POINTER_RESTRICT grad,
    double* POINTER_RESTRICT det_j)
{
  constexpr int dim = 2;

  double dx_ds1 = 0.0;  double dx_ds2 = 0.0;
  double dy_ds1 = 0.0;  double dy_ds2 = 0.0;

  //compute Jacobian
  for (int node = 0; node < nodesPerElement; ++node) {
    const int vector_offset = dim * node;

    const double xCoord = elemNodalCoords[vector_offset + 0];
    const double yCoord = elemNodalCoords[vector_offset + 1];
    const double dn_ds1 = shapeDeriv[vector_offset + 0];
    const double dn_ds2 = shapeDeriv[vector_offset + 1];

    dx_ds1 += dn_ds1 * xCoord;
    dx_ds2 += dn_ds2 * xCoord;

    dy_ds1 += dn_ds1 * yCoord;
    dy_ds2 += dn_ds2 * yCoord;
  }

  *det_j = dx_ds1*dy_ds2 - dy_ds1*dx_ds2;

  const double inv_det_j = 1.0 / (*det_j);

  const double ds1_dx =  inv_det_j*dy_ds2;
  const double ds2_dx = -inv_det_j*dy_ds1;

  const double ds1_dy = -inv_det_j*dx_ds2;
  const double ds2_dy =  inv_det_j*dx_ds1;

  for (int node = 0; node < nodesPerElement; ++node) {
    const int vector_offset = dim * node;

    const double dn_ds1 = shapeDeriv[vector_offset + 0];
    const double dn_ds2 = shapeDeriv[vector_offset + 1];

    grad[vector_offset + 0] = dn_ds1 * ds1_dx + dn_ds2 * ds2_dx;
    grad[vector_offset + 1] = dn_ds1 * ds1_dy + dn_ds2 * ds2_dy;
  }
}

void gradient_3d(
    int nodesPerElement,
    const double* POINTER_RESTRICT elemNodalCoords,
    const double* POINTER_RESTRICT shapeDeriv,
    double* POINTER_RESTRICT grad,
    double* POINTER_RESTRICT det_j)
{
  constexpr int dim = 3;

  double dx_ds1 = 0.0;  double dx_ds2 = 0.0; double dx_ds3 = 0.0;
  double dy_ds1 = 0.0;  double dy_ds2 = 0.0; double dy_ds3 = 0.0;
  double dz_ds1 = 0.0;  double dz_ds2 = 0.0; double dz_ds3 = 0.0;

  //compute Jacobian
  int vector_offset = 0;
  for (int node = 0; node < nodesPerElement; ++node) {
    const double xCoord = elemNodalCoords[vector_offset + 0];
    const double yCoord = elemNodalCoords[vector_offset + 1];
    const double zCoord = elemNodalCoords[vector_offset + 2];

    const double dn_ds1 = shapeDeriv[vector_offset + 0];
    const double dn_ds2 = shapeDeriv[vector_offset + 1];
    const double dn_ds3 = shapeDeriv[vector_offset + 2];

    dx_ds1 += dn_ds1 * xCoord;
    dx_ds2 += dn_ds2 * xCoord;
    dx_ds3 += dn_ds3 * xCoord;

    dy_ds1 += dn_ds1 * yCoord;
    dy_ds2 += dn_ds2 * yCoord;
    dy_ds3 += dn_ds3 * yCoord;

    dz_ds1 += dn_ds1 * zCoord;
    dz_ds2 += dn_ds2 * zCoord;
    dz_ds3 += dn_ds3 * zCoord;

    vector_offset += dim;
  }

  *det_j = dx_ds1 * ( dy_ds2 * dz_ds3 - dz_ds2 * dy_ds3 )
         + dy_ds1 * ( dz_ds2 * dx_ds3 - dx_ds2 * dz_ds3 )
         + dz_ds1 * ( dx_ds2 * dy_ds3 - dy_ds2 * dx_ds3 );

  const double inv_det_j = 1.0 / (*det_j);

  const double ds1_dx = inv_det_j*(dy_ds2 * dz_ds3 - dz_ds2 * dy_ds3);
  const double ds2_dx = inv_det_j*(dz_ds1 * dy_ds3 - dy_ds1 * dz_ds3);
  const double ds3_dx = inv_det_j*(dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2);

  const double ds1_dy = inv_det_j*(dz_ds2 * dx_ds3 - dx_ds2 * dz_ds3);
  const double ds2_dy = inv_det_j*(dx_ds1 * dz_ds3 - dz_ds1 * dx_ds3);
  const double ds3_dy = inv_det_j*(dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2);

  const double ds1_dz = inv_det_j*(dx_ds2 * dy_ds3 - dy_ds2 * dx_ds3);
  const double ds2_dz = inv_det_j*(dy_ds1 * dx_ds3 - dx_ds1 * dy_ds3);
  const double ds3_dz = inv_det_j*(dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2);

  // metrics
  vector_offset = 0;
  for (int node = 0; node < nodesPerElement; ++node) {
    const double dn_ds1 = shapeDeriv[vector_offset + 0];
    const double dn_ds2 = shapeDeriv[vector_offset + 1];
    const double dn_ds3 = shapeDeriv[vector_offset + 2];

    grad[vector_offset + 0] = dn_ds1 * ds1_dx + dn_ds2 * ds2_dx + dn_ds3 * ds3_dx;
    grad[vector_offset + 1] = dn_ds1 * ds1_dy + dn_ds2 * ds2_dy + dn_ds3 * ds3_dy;
    grad[vector_offset + 2] = dn_ds1 * ds1_dz + dn_ds2 * ds2_dz + dn_ds3 * ds3_dz;

    vector_offset += dim;
  }
}

double parametric_distance_quad(const double* x)
{
  double absXi  = std::abs(x[0]);
  double absEta = std::abs(x[1]);
  return (absXi > absEta) ? absXi : absEta;
}

double parametric_distance_hex(const double* x)
{
  std::array<double, 3> y;
  for (int i=0; i<3; ++i) {
    y[i] = std::fabs(x[i]);
  }

  double d = 0;
  for (int i=0; i<3; ++i) {
    if (d < y[i]) {
      d = y[i];
    }
  }
  return d;
}

KOKKOS_FUNCTION
int ip_per_face(const TensorProductQuadratureRule& quad, const LagrangeBasis& basis) {
  return quad.num_quad() * quad.num_quad() * (basis.order() + 1)*(basis.order() + 1);
}

} // namespace nalu
} // namespace Sierra