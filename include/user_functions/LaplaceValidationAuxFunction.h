/* 
 * File:   LaplaceValidationAuxFunction.h
 * Author: Raphael Lindegger
 *
 * Created on January 4, 2020, 2:09 PM
 */

#ifndef LAPLACEVALIDATIONAUXFUNCTION_H
#define LAPLACEVALIDATIONAUXFUNCTION_H

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class LaplaceValidationAuxFunction : public AuxFunction
{
public:

  LaplaceValidationAuxFunction();

  virtual ~LaplaceValidationAuxFunction() {}
  
  using AuxFunction::do_evaluate;
  virtual void do_evaluate(
    const double * coords,
    const double time,
    const unsigned spatialDimension,
    const unsigned numPoints,
    double * fieldPtr,
    const unsigned fieldSize,
    const unsigned beginPos,
    const unsigned endPos) const;
  
private:
  double pi_;
};

} // namespace nalu
} // namespace Sierra

#endif /* LAPLACEVALIDATIONAUXFUNCTION_H */

