/* 
 * File:   cfdVector.h
 * Author: Raphael Lindegger
 *
 * Created on September 25, 2019, 10:44 AM
 */

#ifndef CFDVECTOR_H
#define CFDVECTOR_H

#include <cstdlib>
#include <iostream>
#include <cmath>

namespace sierra {
namespace nalu {

typedef double scalar;
typedef int label;

/**
 * Vector class to make some nice linear algebra expressions 
 * within algorithms
 * Class is designed to handle 2D and 3D vectors.
 */
class cfdVector {
public:
    cfdVector() {}
    
    cfdVector(int size) {
        size_ = size;
        for (int i = 0; i < 3; i++) {
            data_[i] = 0;
        }
    }
    
    cfdVector(scalar x, scalar y) {
        size_ = 2;
        data_[0] = x;
        data_[1] = y;
    }
    
    cfdVector(scalar x, scalar y, scalar z) {
        size_ = 3;
        data_[0] = x;
        data_[1] = y;
        data_[2] = z;
    }
    
    ~cfdVector() {}
    
    inline void print() {
        for (int i = 0; i < size_; i++) {
            std::cout << data_[i] << ", ";
        }
        std::cout << std::endl;
    }
    
    inline void set_size(int size) {
        size_ = size;
    }
    
    inline int get_size() {
        return size_;
    }
    
    inline void set(const scalar x, const scalar y) {
        size_ = 2;
        data_[0] = x;
        data_[1] = y;
    }
    
    inline void set(const scalar x, const scalar y, const scalar z) {
        size_ = 3;
        data_[0] = x;
        data_[1] = y;
        data_[2] = z;
    }
    
    inline scalar & operator[](const int idx) {
        return  data_[idx];
    }
    
    inline const scalar & operator[](const label idx) const {
        return data_[idx];
    }
    
    inline cfdVector & operator=(const scalar r) {
        for (label i = 0; i < size_; i++) {
            data_[i] = r;
        }
        return *this;
    }
    
    inline cfdVector & operator=(const cfdVector v) {
        for (label i = 0; i < v.size_; i++) {
            data_[i] = v[i];
        }
        return *this;
    }

    inline cfdVector operator+(const cfdVector & v) const {
        cfdVector result(v.size_);
        for (label i = 0; i < size_; i++) {
            result[i] = data_[i] + v[i];
        }
        return result;
    }
    
    inline cfdVector & operator+=(const cfdVector & v) {
        for (label i = 0; i < size_; i++) {
            data_[i] += v[i];
        }
        return *this;
    }
    
    inline cfdVector operator-(const cfdVector & v) const {
        cfdVector result(v.size_);
        for (int i = 0; i < size_; i++) {
            result[i] = data_[i] - v[i];
        }
        return result;
    }
    
    inline cfdVector operator-() const {
        cfdVector result(size_);
        for (int i = 0; i < size_; i++) {
            result[i] = -data_[i];
        }
        return result;
    }
    
    inline cfdVector & operator-=(const cfdVector & v) {
        for (int i = 0; i < size_; i++) {
            data_[i] -= v[i];
        }
        return *this;
    }
    
    // Product vector * scalar
    inline friend cfdVector operator*(const cfdVector & v, const scalar r) {
        cfdVector result(v.size_);
        for (int i = 0; i < result.size_; i++) {
            result[i] =  v[i] * r;
        }
        return result;
    }
    
    // Product scalar * vector
    inline friend cfdVector operator*(const scalar r, const cfdVector & v) {
        cfdVector result(v.size_);
        for (int i = 0; i < result.size_; i++) {
            result[i] =  v[i] * r;
        }
        return result;
    }
    
    inline cfdVector operator/(const scalar r) const {
        cfdVector result(size_);
        for (int i = 0; i < size_; i++) {
            result[i] =  data_[i] / r;
        }
        return result;
    }
    
    inline cfdVector & operator*=(const scalar r) {
        for (int i = 0; i < size_; i++) {
            data_[i] *= r;
        }
        return *this;
    }
    
    inline cfdVector & operator/=(const scalar r) {
        for (int i = 0; i < size_; i++) {
            data_[i] /= r;
        }
        return *this;
    }
    
    // Magnitude
    inline friend scalar magnitude(const cfdVector & v) {
        scalar tmp = 0;
        for (int i = 0; i < v.size_; i++) {
            tmp += v[i]*v[i];
        }
        return sqrt(tmp);
    }
    
    // Dot product
    inline friend scalar operator&(const cfdVector & v1, const cfdVector & v2) {
        scalar result = 0;
        for (int i = 0; i < v1.size_; i++) {
            result +=  v1[i] * v2[i];
        }   
        return result;
    }
    
    // Cross product, only implemented for vector size = 3;
    inline friend cfdVector operator^(const cfdVector & v1,const cfdVector & v2) {
        if (v1.size_ != 3) {
            throw std::runtime_error("cross product not defined for this vector size");
        }
        return cfdVector(v1.data_[1]*v2.data_[2] - v1.data_[2]*v2.data_[1],
                         v1.data_[2]*v2.data_[0] - v1.data_[0]*v2.data_[2],
                         v1.data_[0]*v2.data_[1] - v1.data_[1]*v2.data_[0]);
    }
    
private:
    scalar data_[3];
    int size_;
};

} // namespace nalu
} // namespace Sierra

#endif /* CFDVECTOR_H */

