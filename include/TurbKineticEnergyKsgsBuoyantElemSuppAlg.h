/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef TurbKineticEnergyKsgsBuoyantElemSuppAlg_h
#define TurbKineticEnergyKsgsBuoyantElemSuppAlg_h

#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>

namespace sierra{
namespace nalu{

class Realm;
class MasterElement;

class TurbKineticEnergyKsgsBuoyantElemSuppAlg : public SupplementalAlgorithm
{
public:

  TurbKineticEnergyKsgsBuoyantElemSuppAlg(
    Realm &realm);

  virtual ~TurbKineticEnergyKsgsBuoyantElemSuppAlg() {}

  virtual void setup();

  virtual void elem_resize(
    MasterElement *meSCS,
    MasterElement *meSCV);

  virtual void elem_execute(
    double *lhs,
    double *rhs,
    stk::mesh::Entity element,
    MasterElement *meSCS,
    MasterElement *meSCV);

  double cross_product_magnitude();

  const stk::mesh::BulkData *bulkData_;

  ScalarFieldType *tkeNp1_;
  ScalarFieldType *densityNp1_;
  ScalarFieldType *dualNodalVolume_;
  VectorFieldType *coordinates_;

  const double CbTwo_;
  const int nDim_;

  // fixed space
  std::vector<double> ws_gravity_;
  std::vector<double> ws_drhodx_;

  // scratch space
  std::vector<double> ws_dndx_;
  std::vector<double> ws_deriv_;
  std::vector<double> ws_det_j_;
  std::vector<double> ws_shape_function_;

  std::vector<double> ws_tkeNp1_;
  std::vector<double> ws_rhoNp1_;
  std::vector<double> ws_dualNodalVolume_;
  std::vector<double> ws_coordinates_;
  std::vector<double> ws_scv_volume_;
};

} // namespace nalu
} // namespace Sierra

#endif
