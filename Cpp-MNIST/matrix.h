#pragma once

#ifndef __matrix_H
#define __matrix_H

#include <vector>

template <typename T> class matrix {
private:
	std::vector<std::vector<T> > mat;
	unsigned rows;
	unsigned cols;

public:
	matrix(unsigned _rows, unsigned _cols, const T& _initial);
	matrix(const matrix<T>& rhs);
	matrix(const std::vector<std::vector<T> >& rhs);
	matrix(int _rows, int _cols, const std::vector<T>& rhs);
	virtual ~matrix();
	matrix<T> as_vectors() { return mat; }

	// Operator overloading, for "standard" mathematical matrix operations                                                                                                                                                          
	matrix<T>& operator=(const matrix<T>& rhs);

	// Matrix mathematical operations                                                                                                                                                                                               
	matrix<T> operator+(const matrix<T>& rhs);
	matrix<T>& operator+=(const matrix<T>& rhs);
	matrix<T> operator-(const matrix<T>& rhs);
	matrix<T>& operator-=(const matrix<T>& rhs);
	matrix<T> operator*(const matrix<T>& rhs);
	matrix<T>& operator*=(const matrix<T>& rhs);
	matrix<T> operator/(const matrix<T>& rhs);
	matrix<T>& operator/=(const matrix<T>& rhs);
	matrix<T> transpose();
	matrix<T> dot(const matrix<T>& rhs);

	// Matrix/scalar operations                                                                                                                                                                                                     
	matrix<T> operator+(const T& rhs);
	matrix<T>& operator+=(const T& rhs);
	matrix<T> operator-(const T& rhs);
	matrix<T>& operator-=(const T& rhs);
	matrix<T> operator*(const T& rhs);
	matrix<T>& operator*=(const T& rhs);
	matrix<T> operator/(const T& rhs);
	matrix<T>& operator/=(const T& rhs);

	// Matrix/vector operations                        
	matrix<T> operator+(const std::vector<T>& rhs);
	matrix<T>& operator+=(const std::vector<T>& rhs);
	matrix<T> operator-(const std::vector<T>& rhs);
	matrix<T>& operator-=(const std::vector<T>& rhs);
	matrix<T> operator*(const std::vector<T>& rhs);
	matrix<T>& operator*=(const std::vector<T>& rhs);
	matrix<T> operator/(const std::vector<T>& rhs);
	matrix<T>& operator/=(const std::vector<T>& rhs);
	
	std::vector<T> diag_vec();
	float sum();
	//matrix<float> dot(matrix<float> A, matrix<float> B, bool gpu);

	std::vector<T> flatten();
	matrix<T> sum_v();
	matrix<T> sum_h();
	matrix<T> concatenate_v(matrix<T>& rhs);
	matrix<T> concatenate_h(matrix<T>& rhs);
	matrix<T> slice_v(int start_index, int end_index);
	matrix<T> slice_h(int start_index, int end_index);
	matrix<T> maximum(const T& max);
	matrix<T> minimum(const T& min);
	matrix<T> maximum_mat(matrix<T>& rhs);
	matrix<T> minimum_mat(matrix<T>& rhs);
	matrix<T> exponent();

	std::string get_size_str();

	// Access the individual elements                                                                                                                                                                                               
	T& operator()(const unsigned& row, const unsigned& col);
	const T& operator()(const unsigned& row, const unsigned& col) const;

	// Access the individual elements as rows
	std::vector<T>& operator()(const unsigned& row);
	const std::vector<T>& operator()(const unsigned& row) const;

	// Access the row and column sizes                                                                                                                                                                                              
	unsigned get_rows() const;
	unsigned get_cols() const;

};

#include "matrix.cpp"

#endif