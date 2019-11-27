/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporatlion.                                   */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <master_element/HigherOrderQuad2DSCS.h>
#include <master_element/MasterElementFunctions.h>
#include <master_element/HigherOrderMasterElementFunctions.h>
#include <master_element/MasterElementUtils.h>
#include <master_element/TensorOps.h>

#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/LagrangeBasis.h>

#include <NaluEnv.h>
#include <master_element/MasterElement.h>
#include <FORTRAN_Proto.h>

#include <BuildTemplates.h>

#include <stk_util/util/ReportHandler.hpp>

#include <array>
#include <limits>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <element_promotion/ElementDescription.h>

namespace sierra{
namespace nalu{

KOKKOS_FUNCTION
HigherOrderQuad2DSCS::HigherOrderQuad2DSCS(
  LagrangeBasis basis,
  TensorProductQuadratureRule quadrature)
: MasterElement(),
  basis_(std::move(basis)),
  quadrature_(std::move(quadrature))
#ifndef KOKKOS_ENABLE_CUDA
  , nodeMap(make_node_map_quad(basis.order())),
  faceNodeMap(make_face_node_map_quad(basis.order())),
  sideNodeOrdinals_(make_side_node_ordinal_map_quad(basis.order()))
#endif
  , nodes1D_(basis.order()+1)
{
  MasterElement::nDim_ = 2;
  nodesPerElement_ = nodes1D_*nodes1D_;

#ifndef KOKKOS_ENABLE_CUDA
  // set up integration rule and relevant maps for scs
  set_interior_info();

  // set up integration rule and relevant maps for faces
  set_boundary_info();

  // compute and save shape functions and derivatives at ips
  shapeFunctionVals_ = basis_.eval_basis_weights(intgLoc_);
  shapeDerivs_ = basis_.eval_deriv_weights(intgLoc_);
  expFaceShapeDerivs_ = basis_.eval_deriv_weights(intgExpFace_);
#endif
}

double HigherOrderQuad2DSCS::isInElement(
    const double *elemNodalCoord,
    const double *pointCoord,
    double *isoParCoord)
{
  std::array<double, 2> initialGuess = {{ 0.0, 0.0 }};
  int maxIter = 50;
  double tolerance = 1.0e-16;
  double deltaLimit = 1.0e4;

  bool converged = isoparameteric_coordinates_for_point_2d(
      basis_,
      elemNodalCoord,
      pointCoord,
      isoParCoord,
      initialGuess,
      maxIter,
      tolerance,
      deltaLimit
  );
  ThrowAssertMsg(parametric_distance_quad(isoParCoord) < 1.0 + 1.0e-6 || !converged,
      "Inconsistency in parametric distance calculation");

  return (converged) ? parametric_distance_quad(isoParCoord) : std::numeric_limits<double>::max();
}


void HigherOrderQuad2DSCS::interpolatePoint(
  const int &nComp,
  const double *isoParCoord,
  const double *field,
  double *result)
{
  const auto& weights = basis_.point_interpolation_weights(isoParCoord);
  for (int n = 0; n < nComp; ++n) {
    result[n] = ddot(weights.data(), field + n * nodesPerElement_, nodesPerElement_);
  }
}

void
HigherOrderQuad2DSCS::set_interior_info()
{
  const int linesPerDirection = nodes1D_ - 1;
  const int ipsPerLine = quadrature_.num_quad() * nodes1D_;
  const int numLines = linesPerDirection * nDim_;

  numIntPoints_ = numLines * ipsPerLine;

  // define L/R mappings
  lrscv_ = Kokkos::View<int*>("lsrcv_", 2*numIntPoints_);

  // standard integration location
  intgLoc_ =  Kokkos::View<double*[2]>("intgLoc_", numIntPoints_);
  ipWeights_ = Kokkos::View<double*>("ipWeights_",numIntPoints_);

  // specify integration point locations in a dimension-by-dimension manner

  //u-direction
//  std::cout << "### u-direction" << std::endl;
  int lrscv_index = 0;
  int scalar_index = 0;
  for (int m = 0; m < linesPerDirection; ++m) {
    for (int l = 0; l < nodes1D_; ++l) {

      int leftNode;
      int rightNode;
      int orientation;
      if (m % 2 == 0) {
        leftNode  = nodeMap(m + 0, l);
        rightNode = nodeMap(m + 1, l);
        orientation = -1;
      }
      else {
        leftNode  = nodeMap(m + 1, l);
        rightNode = nodeMap(m + 0, l);
        orientation = +1;
      }

      for (int j = 0; j < quadrature_.num_quad(); ++j) {

        lrscv_(lrscv_index) = leftNode;
        lrscv_(lrscv_index + 1) = rightNode;

        intgLoc_(scalar_index, 0) = quadrature_.integration_point_location(l,j);
        intgLoc_(scalar_index, 1) = quadrature_.scs_loc(m);
//        std::cout << "intgLoc_[*,0]: " << quadrature_.integration_point_location(l,j) << std::endl;
//        std::cout << "intgLoc_[*,1]: " << quadrature_.scs_loc(m) << std::endl;

        //compute the quadrature weight
        ipWeights_(scalar_index) = orientation*quadrature_.integration_point_weight(l,j);
//        std::cout << "ipWeights_: " << orientation*quadrature_.integration_point_weight(l,j) << std::endl;

        ++scalar_index;
        lrscv_index += 2;
      }
    }
  }

  //t-direction
//  std::cout << "### t-direction" << std::endl;
  for (int m = 0; m < linesPerDirection; ++m) {
    for (int l = 0; l < nodes1D_; ++l) {

      int leftNode;
      int rightNode;
      int orientation;
      if (m % 2 == 0) {
        leftNode  = nodeMap(l,m);
        rightNode = nodeMap(l,m+1);
        orientation = +1;
      }
      else {
        leftNode  = nodeMap(l,m+1);
        rightNode = nodeMap(l,m);
        orientation = -1;
      }

      for (int j = 0; j < quadrature_.num_quad(); ++j) {

        lrscv_(lrscv_index)   = leftNode;
        lrscv_(lrscv_index+1) = rightNode;

        intgLoc_(scalar_index, 0) = quadrature_.scs_loc(m);
        intgLoc_(scalar_index, 1) = quadrature_.integration_point_location(l,j);
//        std::cout << "intgLoc_[*,0]: " << quadrature_.scs_loc(m) << std::endl;
//        std::cout << "intgLoc_[*,1]: " << quadrature_.integration_point_location(l,j) << std::endl;

        //compute the quadrature weight
        ipWeights_(scalar_index) = orientation*quadrature_.integration_point_weight(l,j);
//        std::cout << "ipWeights_: " << orientation*quadrature_.integration_point_weight(l,j) << std::endl;
        
        ++scalar_index;
        lrscv_index += 2;
      }
    }
  }
}

void
HigherOrderQuad2DSCS::set_boundary_info()
{
  const int numFaces = 2*nDim_;
  const int nodesPerFace = nodes1D_;
  const int linesPerDirection = nodes1D_-1;
  ipsPerFace_ = nodesPerFace*quadrature_.num_quad();

  const int numFaceIps = numFaces*ipsPerFace_;

  oppFace_ =Kokkos::View<int*>("oppFace_", numFaceIps);
  ipNodeMap_ = Kokkos::View<int*>("ipNodeMap_",numFaceIps);
  oppNode_ = Kokkos::View<int*>("oppNode", numFaceIps);
  intgExpFace_=Kokkos::View<double**>("intgExpFace_",numFaceIps,nDim_);


  auto face_node_number = [&] (int number,int faceOrdinal)
  {
    return faceNodeMap(faceOrdinal,number);
  };

  const std::array<int, 4> faceToLine = {{
      0,
      2*linesPerDirection-1,
      linesPerDirection-1,
      linesPerDirection
  }};

  const std::array<double, 4> faceLoc = {{-1.0, +1.0, +1.0, -1.0}};

  int scalar_index = 0;
  int faceOrdinal = 0; //bottom face
  int oppFaceIndex = 0;
  for (int k = 0; k < nodes1D_; ++k) {
    const int nearNode = face_node_number(k,faceOrdinal);
    int oppNode = nodeMap(k, 1);
//    std::cout << "nearNode: " << nearNode << std::endl;
//    std::cout << "oppNode: " << oppNode << std::endl;

    for (int j = 0; j < quadrature_.num_quad(); ++j) {
      ipNodeMap_(scalar_index) = nearNode;
      oppNode_(scalar_index) = oppNode;
      oppFace_(scalar_index) = oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_;

      intgExpFace_(scalar_index, 0) = intgLoc_(oppFace_(scalar_index), 0);
//      std::cout << "intgExpFace_(" << scalar_index << ", " << 0 << ") = " << "intgLoc_(oppFace_(" << scalar_index << "), " << 0 << "); = " << intgExpFace_(scalar_index, 0) << std::endl;
      intgExpFace_(scalar_index, 1) = faceLoc[faceOrdinal];
//      std::cout << "intgExpFace_(" << scalar_index << ", " << 1 << ") = " << "faceLoc[" << faceOrdinal << "]; = " << intgExpFace_(scalar_index, 1) << std::endl;

      ++scalar_index;
      ++oppFaceIndex;
    }
  }

  faceOrdinal = 1; //right face
  oppFaceIndex = 0;
  for (int k = 0; k < nodes1D_; ++k) {
    const int nearNode = face_node_number(k,faceOrdinal);
    int oppNode = nodeMap(k, nodes1D_-2);

    for (int j = 0; j < quadrature_.num_quad(); ++j) {
      ipNodeMap_(scalar_index) = nearNode;
      oppNode_(scalar_index) = oppNode;
      oppFace_(scalar_index) = oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_;

      intgExpFace_(scalar_index, 0) = faceLoc[faceOrdinal];
      intgExpFace_(scalar_index, 1) = intgLoc_(oppFace_(scalar_index), 1);

      ++scalar_index;
      ++oppFaceIndex;
    }
  }

  faceOrdinal = 2; //top face
  oppFaceIndex = 0;
  //NOTE: this face is reversed
  int elemNodeM1 = static_cast<int>(nodes1D_-1);
  for (int k = elemNodeM1; k >= 0; --k) {
    const int nearNode = face_node_number(nodes1D_-k-1,faceOrdinal);
    int oppNode = nodeMap(nodes1D_-2, k);
    for (int j = 0; j < quadrature_.num_quad(); ++j) {
      ipNodeMap_(scalar_index) = nearNode;
      oppNode_(scalar_index) = oppNode;
      oppFace_(scalar_index) = (ipsPerFace_-1) - oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_;

      intgExpFace_(scalar_index, 0) = intgLoc_(oppFace_(scalar_index), 0);
      intgExpFace_(scalar_index, 1) = faceLoc[faceOrdinal];

      ++scalar_index;
      ++oppFaceIndex;
    }
  }

  faceOrdinal = 3; //left face
  oppFaceIndex = 0;
  //NOTE: this face is reversed
  for (int k = elemNodeM1; k >= 0; --k) {
    const int nearNode = face_node_number(nodes1D_-k-1,faceOrdinal);
    int oppNode = nodeMap(k,1);
    for (int j = 0; j < quadrature_.num_quad(); ++j) {
      ipNodeMap_(scalar_index) = nearNode;
      oppNode_(scalar_index) = oppNode;
      oppFace_(scalar_index) = (ipsPerFace_-1) - oppFaceIndex + faceToLine[faceOrdinal]*ipsPerFace_;

      intgExpFace_(scalar_index, 0)   = faceLoc[faceOrdinal];
      intgExpFace_(scalar_index, 1) = intgLoc_(oppFace_(scalar_index), 1);

      ++scalar_index;
      ++oppFaceIndex;
    }
  }
}

void
HigherOrderQuad2DSCS::shape_fcn(double *shpfc)
{
  int numShape = shapeFunctionVals_.size();
  for (int j = 0; j < numShape; ++j) {
    shpfc[j] = shapeFunctionVals_.data()[j];
  }
}

const int *
HigherOrderQuad2DSCS::ipNodeMap(int ordinal) const
{
  // define ip->node mappings for each face (ordinal);
  return &ipNodeMap_(ordinal*ipsPerFace_);
}

const int *
HigherOrderQuad2DSCS::side_node_ordinals(int ordinal) const
{
  return &sideNodeOrdinals_(ordinal,0);
}

void
HigherOrderQuad2DSCS::determinant(
  const int nelem,
  const double *coords,
  double *areav,
  double *error)
{
  //returns the normal vector (dyds,-dxds) for constant t curves
  //returns the normal vector (dydt,-dxdt) for constant s curves
  *error = 0.0;
  ThrowRequireMsg(nelem == 1, "determinant is executed one element at a time for HO");

  constexpr int dim = 2;
  int ipsPerDirection = numIntPoints_ / dim;

  int index = 0;

   //returns the normal vector x_u x x_s for constant t surfaces
  for (int ip = 0; ip < ipsPerDirection; ++ip) {
    area_vector<Jacobian::T_DIRECTION>(coords, &shapeDerivs_(index,0,0), &areav[index*dim]);
    ++index;
  }

  //returns the normal vector x_t x x_u for constant s curves
  for (int ip = 0; ip < ipsPerDirection; ++ip) {
    area_vector<Jacobian::S_DIRECTION>(coords, &shapeDerivs_(index,0,0), &areav[index*dim]);
    ++index;
  }

  // Multiply with the integration point weighting
  for (int ip = 0; ip < numIntPoints_; ++ip) {

    double weight = ipWeights_(ip);
    areav[ip * dim + 0] *= weight;
    areav[ip * dim + 1] *= weight;
    
//    std::cout << "IP: " << ip+1 << std::endl;
//    std::cout << "areav_x: " << areav[ip * dim + 0] << std::endl;
//    std::cout << "areav_y: " << areav[ip * dim + 1] << std::endl;
  }
}

void HigherOrderQuad2DSCS::grad_op(
  const int nelem,
  const double *coords,
  double *gradop,
  double *deriv,
  double *det_j,
  double *error)
{
  *error = 0.0;
  ThrowRequireMsg(nelem == 1, "Grad_op is executed one element at a time for HO");

  int grad_offset = 0;
  int grad_inc = nDim_ * nodesPerElement_;

  for (int ip = 0; ip < numIntPoints_; ++ip) {
    for (int j = 0; j < grad_inc; ++j) {
      deriv[grad_offset + j] = shapeDerivs_.data()[grad_offset +j];
    }

    gradient_2d(
      nodesPerElement_,
      &coords[0],
      &shapeDerivs_.data()[grad_offset],
      &gradop[grad_offset],
      &det_j[ip]
    );

    if (det_j[ip] < tiny_positive_value()) {
      *error = 1.0;
    }

    grad_offset += grad_inc;
  }
}

void
HigherOrderQuad2DSCS::face_grad_op(
  const int nelem,
  const int face_ordinal,
  const double *coords,
  double *gradop,
  double *det_j,
  double *error)
{
  *error = 0.0;
  ThrowRequireMsg(nelem == 1, "face_grad_op is executed one element at a time for HO");

  int grad_offset = 0;
  int grad_inc = nDim_ * nodesPerElement_;

  const int face_offset =  nDim_ * ipsPerFace_ * nodesPerElement_ * face_ordinal;
  const double* const faceShapeDerivs = &expFaceShapeDerivs_.data()[face_offset];

  for (int ip = 0; ip < ipsPerFace_; ++ip) {
    gradient_2d(
      nodesPerElement_,
      coords,
      &faceShapeDerivs[grad_offset],
      &gradop[grad_offset],
      &det_j[ip]
   );

    if (det_j[ip] < tiny_positive_value()) {
      *error = 1.0;
    }

    grad_offset += grad_inc;
  }
}


const int *
HigherOrderQuad2DSCS::adjacentNodes()
{
  // define L/R mappings
  return lrscv_.data();
}

int
HigherOrderQuad2DSCS::opposingNodes(
  const int ordinal,
  const int node)
{
  return oppNode_(ordinal*ipsPerFace_+node);
}

int
HigherOrderQuad2DSCS::opposingFace(
  const int ordinal,
  const int node)
{
  return oppFace_(ordinal*ipsPerFace_+node);
}

template <Jacobian::Direction direction> void
HigherOrderQuad2DSCS::area_vector(
  const double *POINTER_RESTRICT elemNodalCoords,
  double *POINTER_RESTRICT shapeDeriv,
  double *POINTER_RESTRICT normalVec ) const
{
  constexpr int s1Component = (direction == Jacobian::S_DIRECTION) ?
      Jacobian::T_DIRECTION : Jacobian::S_DIRECTION;

  double dxdr = 0.0;  double dydr = 0.0;
  for (int node = 0; node < nodesPerElement_; ++node) {
    const int vector_offset = nDim_ * node;
    const double xCoord = elemNodalCoords[vector_offset+0];
    const double yCoord = elemNodalCoords[vector_offset+1];

    dxdr += shapeDeriv[vector_offset+s1Component] * xCoord;
    dydr += shapeDeriv[vector_offset+s1Component] * yCoord;
  }

  normalVec[0] =  dydr;
  normalVec[1] = -dxdr;
}

void HigherOrderQuad2DSCS::gij(
  const double *coords,
  double *gupperij,
  double *glowerij,
  double *deriv)
{
  SIERRA_FORTRAN(twod_gij)
    ( &nodesPerElement_,
      &numIntPoints_,
      deriv,
      coords, gupperij, glowerij);
}

}  // namespace nalu
} // namespace sierra
