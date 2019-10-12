#include "pch.h"

#ifndef __matrix_CPP
#define __matrix_CPP

#include "matrix.h"
#include <iostream>
#include <sstream>
#include <cmath>


// Parameter Constructor                                                                                                                                                      
template<typename T>
matrix<T>::matrix(unsigned _rows, unsigned _cols, const T& _initial)
{
	mat.resize(_rows);
	for (unsigned i = 0; i < mat.size(); i++)
	{
		mat[i].resize(_cols, _initial);
	}
	rows = _rows;
	cols = _cols;
}

// Copy Constructor                                                                                                                                                           
template<typename T>
matrix<T>::matrix(const matrix<T>& rhs)
{
	mat = rhs.mat;
	rows = rhs.get_rows();
	cols = rhs.get_cols();
}

template<typename T>
matrix<T>::matrix(const std::vector<std::vector<T> >& rhs)
{
	mat.resize(rhs.size());
	for (unsigned i = 0; i < mat.size(); i++)
	{
		mat[i].resize(rhs[i].size());
	}
	mat = rhs;
	rows = rhs.size();
	cols = rhs[0].size();
}

// create from 1D vector
template<typename T>
matrix<T>::matrix(int _rows, int _cols, const std::vector<T>& rhs)
{
	mat.resize(_rows);
	unsigned index = 0;
	for (unsigned i = 0; i < _rows; i++)
	{
		mat[i].resize(_cols);
		for (unsigned j = 0; j < _cols; j++)
		{
			mat[i][j] = rhs[index];
			index++;
		}
	}
	rows = _rows;
	cols = _cols;
}

// (Virtual) Destructor                                                                                                                                                       
template<typename T>
matrix<T>::~matrix() {}

// Assignment Operator with matrix                                                                                                                 
template<typename T>
matrix<T>& matrix<T>::operator=(const matrix<T>& rhs) {
	if (&rhs == this)
		return *this;

	unsigned new_rows = rhs.get_rows();
	unsigned new_cols = rhs.get_cols();

	mat.resize(new_rows);
	for (unsigned i = 0; i < mat.size(); i++) {
		mat[i].resize(new_cols);
	}

	for (unsigned i = 0; i < new_rows; i++) {
		for (unsigned j = 0; j < new_cols; j++) {
			mat[i][j] = rhs(i, j);
		}
	}
	rows = new_rows;
	cols = new_cols;

	return *this;
}


// ----------------------------- Matrix/matrix -------------------------- //


// Addition of two matrices                                                                                                                                                   
template<typename T>
matrix<T> matrix<T>::operator+(const matrix<T>& rhs) {
	if (cols != rhs.get_cols() ||rows != rhs.get_rows())
	{
		std::cout << "Matricies don't align: " <<rows << ", " << cols << " + " << rhs.get_rows() << ", " << rhs.get_cols() << std::endl;
	}
	matrix result(rows, cols, 0.0);

	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			result(i, j) = this->mat[i][j] + rhs(i, j);
		}
	}

	return result;
}

// Cumulative addition of this matrix and another                                                                                                                             
template<typename T>
matrix<T>& matrix<T>::operator+=(const matrix<T>& rhs) {
	if (cols != rhs.get_cols() ||rows != rhs.get_rows())
	{
		std::cout << "Matricies don't align: " <<rows << ", " << cols << " += " << rhs.get_rows() << ", " << rhs.get_cols() << std::endl;
	}

	for (unsigned i = 0; i < rhs.get_rows(); i++) {
		for (unsigned j = 0; j < rhs.get_cols(); j++) {
			this->mat[i][j] += rhs(i, j);
		}
	}

	return *this;
}

// Subtraction of this matrix and another                                                                                                                                     
template<typename T>
matrix<T> matrix<T>::operator-(const matrix<T>& rhs) {

	if (cols != rhs.get_cols() ||rows != rhs.get_rows())
	{
		std::cout << "Matricies don't align: " << rows << ", " << cols << " - " << rhs.get_rows() << ", " << rhs.get_cols() << std::endl;
	}

	matrix result(rhs.get_rows(), rhs.get_cols(), 0.0);

	for (unsigned i = 0; i < rhs.get_rows(); i++) {
		for (unsigned j = 0; j < rhs.get_cols(); j++) {
			result(i, j) = this->mat[i][j] - rhs(i, j);
		}
	}

	return result;
}

// Cumulative subtraction of this matrix and another                                                                                                                          
template<typename T>
matrix<T>& matrix<T>::operator-=(const matrix<T>& rhs) {
	if (cols != rhs.get_cols() ||rows != rhs.get_rows())
	{
		std::cout << "Matricies don't align: " <<rows << ", " << cols << " -= " << rhs.get_rows() << ", " << rhs.get_cols() << std::endl;
	}

	for (unsigned i = 0; i < rhs.get_rows(); i++) {
		for (unsigned j = 0; j < rhs.get_cols(); j++) {
			this->mat[i][j] -= rhs(i, j);
		}
	}

	return *this;
}


// Left multiplication of this matrix and another                                                                                                                              
template<typename T>
matrix<T> matrix<T>::operator*(const matrix<T>& rhs)
{
	if (cols != rhs.get_cols() || rows != rhs.get_rows())
	{
		std::cout << "Matricies don't align: " << rows << ", " << cols << " * " << rhs.get_rows() << ", " << rhs.get_cols() << std::endl;
	}
	matrix result(rows, cols, 0.0);

	for (unsigned i = 0; i < rows; i++)
	{
		for (unsigned j = 0; j < cols; j++)
		{
			result(i, j) = this->mat[i][j] * rhs(i, j);
		}
	}

	return result;
}

// Cumulative left multiplication of this matrix and another                                                                                                                  
template<typename T>
matrix<T>& matrix<T>::operator*=(const matrix<T>& rhs) {
	if (cols != rhs.get_cols() || rows != rhs.get_rows())
	{
		std::cout << "Matricies don't align: " << rows << ", " << cols << " *= " << rhs.get_rows() << ", " << rhs.get_cols() << std::endl;
	}
	matrix result = (*this) * rhs;
	(*this) = result;
	return *this;
}

// Left multiplication of this matrix and another                                                                                                                              
template<typename T>
matrix<T> matrix<T>::operator/(const matrix<T>& rhs)
{
	if (cols != rhs.get_cols() || rows != rhs.get_rows())
	{
		std::cout << "Matricies don't align: " << rows << ", " << cols << " / " << rhs.get_rows() << ", " << rhs.get_cols() << std::endl;
	}
	matrix result(rows, cols, 0.0);

	for (unsigned i = 0; i < rows; i++)
	{
		for (unsigned j = 0; j < cols; j++)
		{
			result(i, j) = this->mat[i][j] / rhs(i, j);
		}
	}

	return result;
}

// Cumulative left division of this matrix and another                                                                                                                  
template<typename T>
matrix<T>& matrix<T>::operator/=(const matrix<T>& rhs) {
	if (cols != rhs.get_cols() || rows != rhs.get_rows())
	{
		std::cout << "Matricies don't align: " << rows << ", " << cols << " /= " << rhs.get_rows() << ", " << rhs.get_cols() << std::endl;
	}
	matrix result = (*this) / rhs;
	(*this) = result;
	return *this;
}

// Calculate a transpose of this matrix                                                                                                                                       
template<typename T>
matrix<T> matrix<T>::transpose() {
	matrix result(cols, rows, 0.0);

	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			result(j, i) = this->mat[i][j];
		}
	}

	return result;
}

// Left multiplication of this matrix and another                                                                                                                              
template<typename T>
matrix<T> matrix<T>::dot(const matrix<T>& rhs)
{
	if (cols != rhs.get_rows())
	{
		std::cout << "Matricies must align: " << rows << ", " << cols << " dot " << rhs.get_rows() << ", " << rhs.get_cols() << std::endl;
	}
	unsigned new_rows = rows;
	unsigned new_cols = rhs.get_cols();
	matrix result(new_rows, new_cols, 0.0);

	for (unsigned i = 0; i < new_rows; i++)
	{
		for (unsigned j = 0; j < new_cols; j++)
		{
			for (unsigned k = 0; k < rhs.get_rows(); k++)
			{
				result(i, j) += mat[i][k] * rhs(k, j);
			}
		}
	}

	return result;
}


// ----------------------------- Matrix/scalar -------------------------- //


// Matrix/scalar addition                                                                                                                                                     
template<typename T>
matrix<T> matrix<T>::operator+(const T& rhs) {
	matrix result(rows, cols, 0.0);

	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			result(i, j) = this->mat[i][j] + rhs;
		}
	}

	return result;
}

// Matrix/scalar addition                                                                                                                                                     
template<typename T>
matrix<T>& matrix<T>::operator+=(const T& rhs) {
	matrix result = (*this) + rhs;
	(*this) = result;
	return *this;
}

// Matrix/scalar subtraction                                                                                                                                                  
template<typename T>
matrix<T> matrix<T>::operator-(const T& rhs) {
	matrix result(rows, cols, 0.0);

	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			result(i, j) = this->mat[i][j] - rhs;
		}
	}

	return result;
}

// Matrix/scalar subtraction                                                                                                                                                  
template<typename T>
matrix<T>& matrix<T>::operator-=(const T& rhs) {
	matrix result = (*this) - rhs;
	(*this) = result;
	return *this;
}

// Matrix/scalar multiplication                                                                                                                                               
template<typename T>
matrix<T> matrix<T>::operator*(const T& rhs) {
	matrix result(mat);

	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			result(i, j) = this->mat[i][j] * rhs;
		}
	}

	return result;
}

// Matrix/scalar multiplication                                                                                                                                               
template<typename T>
matrix<T>& matrix<T>::operator*=(const T& rhs) {
	matrix result = (*this) * rhs;
	(*this) = result;
	return *this;
}

// Matrix/scalar division                                                                                                                                                     
template<typename T>
matrix<T> matrix<T>::operator/(const T& rhs) {
	matrix result(rows, cols, 0.0);

	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			result(i, j) = this->mat[i][j] / rhs;
		}
	}

	return result;
}

// Matrix/scalar division                                                                                                                                                     
template<typename T>
matrix<T>& matrix<T>::operator/=(const T& rhs) {
	matrix result = (*this) / rhs;
	(*this) = result;
	return *this;
}


// ----------------------------- Matrix/vector -------------------------- //


// Add a matrix with a vector                                                                                                                                            
template<typename T>
matrix<T> matrix<T>::operator+(const std::vector<T>& rhs) {
	std::vector<T> result(rhs.size(), 0.0);

	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			result[i] = this->mat[i][j] + rhs[j];
		}
	}

	return result;
}

// Add a vector to a matrix                                                                                                                                        
template<typename T>
matrix<T>& matrix<T>::operator+=(const std::vector<T>& rhs) {

	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			this->mat[i][j] += rhs[j];
		}
	}

	return *this;
}


// Subtract a matrix with a vector                                                                                                                                            
template<typename T>
matrix<T> matrix<T>::operator-(const std::vector<T>& rhs) {
	std::vector<T> result(rhs.size(), 0.0);

	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			result[i] = this->mat[i][j] - rhs[j];
		}
	}

	return result;
}

// Subtract a matrix with a vector                                                                                                                                            
template<typename T>
matrix<T>& matrix<T>::operator-=(const std::vector<T>& rhs) {

	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			this->mat[i][j] -= rhs[j];
		}
	}

	return *this;
}


// Multiply a matrix with a vector                                                                                                                                            
template<typename T>
matrix<T> matrix<T>::operator*(const std::vector<T>& rhs) {
	matrix result(rows, cols, 0.0);

	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			result(i, j) = this->mat[i][j] * rhs[j];
		}
	}

	return result;
}

// Multiply a matrix with a vector                                                                                                                                       
template<typename T>
matrix<T>& matrix<T>::operator*=(const std::vector<T>& rhs) {

	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			this->mat[i][j] *= rhs[j];
		}
	}

	return *this;
}


// Divide a matrix with a vector                                                                                                                                            
template<typename T>
matrix<T> matrix<T>::operator/(const std::vector<T>& rhs) {
	std::vector<T> result(rhs.size(), 0.0);

	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			result[i] = this->mat[i][j] / rhs[j];
		}
	}

	return result;
}

// Divide a matrix with a vector                                                                                                                                            
template<typename T>
matrix<T>& matrix<T>::operator/=(const std::vector<T>& rhs) {

	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			this->mat[i][j] /= rhs[j];
		}
	}

	return *this;
}


// Obtain a vector of the diagonal elements                                                                                                                                   
template<typename T>
std::vector<T> matrix<T>::diag_vec() {
	std::vector<T> result(rows, 0.0);

	for (unsigned i = 0; i < rows; i++) {
		result[i] = this->mat[i][i];
	}

	return result;
}

// Add up everything                                                                                                                                
template<typename T>
float matrix<T>::sum()
{
	float result = 0.0f;

	for (unsigned i = 0; i < rows; i++)
	{
		for (unsigned j = 0; j < cols; j++)
		{
			result += this->mat[i][j];
		}
	}

	return result;
}

// turn into a 1D vector
template<typename T>
std::vector<T> matrix<T>::flatten()
{
	std::vector<T> result(rows * cols, 0);

	int count = 0;
	for (unsigned i = 0; i < rows; i++)
	{
		for (unsigned j = 0; j < cols; j++)
		{
			result[count] = this->mat[i][j];
			count++;
		}
	}

	return result;
}

template<typename T>
matrix<T> matrix<T>::sum_h()
{
	matrix<T> result(rows, 1, 0);

	for (unsigned i = 0; i < rows; i++)
	{
		for (unsigned j = 0; j < cols; j++)
		{
			result(i, 0) += mat[i][j];
		}
	}
	return result;
}

template<typename T>
matrix<T> matrix<T>::sum_v()
{
	matrix<T> result(1, cols, 0);

	for (unsigned i = 0; i < rows; i++)
	{
		for (unsigned j = 0; j < cols; j++)
		{
			result(0, j) += mat[i][j];
		}
	}
	return result;
}

template<typename T>
matrix<T> matrix<T>::concatenate_v(matrix<T>& rhs)
{
	if (rows != rhs.get_rows())
		std::cout << "Matricies must have the same number of rows: " <<rows << ", " << cols << " x " << rhs.get_rows() << ", " << rhs.get_cols() << std::endl;

	matrix<T> result(rows, cols + rhs.get_cols(), 0);

	for (unsigned i = 0; i < rows; i++)
	{
		for (unsigned j = 0; j < cols; j++)
		{
			result(i, j) = this->mat[i][j];
		}
	}
	for (unsigned i = 0; i < rows; i++)
	{
		for (unsigned j = 0; j < rhs.get_cols(); j++)
		{
			result(i, j + rhs.get_cols()) = rhs(i, j);
		}
	}

	return result;
}

template<typename T>
matrix<T> matrix<T>::concatenate_h(matrix<T>& rhs)
{
	if(cols != rhs.get_cols())
		std::cout << "Matricies must have the same number of columns: " <<rows << ", " << cols << " x " << rhs.get_rows() << ", " << rhs.get_cols() << std::endl;

	matrix<T> result(rows + rhs.get_rows(), cols, 0);

	for (unsigned i = 0; i < rows; i++)
	{
		for (unsigned j = 0; j < cols; j++)
		{
			result(i, j) = this->mat[i][j];
		}
	}
	std::cout << "d" << std::endl;
	for (unsigned i = 0; i < rhs.get_rows(); i++)
	{
		for (unsigned j = 0; j < rhs.get_cols(); j++)
		{
			result(i + rhs.get_rows() - 1, j) = rhs(i, j);
		}
	}
	return result;
}

template<typename T>
matrix<T> matrix<T>::slice_v(int start_index, int end_index)
{
	matrix<T> result(end_index - start_index, cols, 0);
	int count = 0;
	for (unsigned i = start_index; i < end_index; i++)
	{
		for (unsigned j = 0; j < cols; j++)
		{
			result(count, j) = this->mat[i][j];
		}
		count++;
	}

	return result;
}
template<typename T>
matrix<T> matrix<T>::slice_h(int start_index, int end_index)
{
	matrix<T> result(rows, end_index - start_index, 0);
	int count = 0;
	for (unsigned j = start_index; j < end_index; j++)
	{
		for (unsigned i = 0; i < rows; i++)
		{
			result(i, count) = this->mat[i][j];
		}
		count++;
	}

	return result;
}


template<typename T>
matrix<T> matrix<T>::maximum(const T& value)
{
	matrix<T> result(rows, cols, 0);
	for (unsigned i = 0; i < rows; i++)
	{
		for (unsigned j = 0; j < cols; j++)
		{
			result(i, j) = std::max(mat[i][j], value);
		}
	}
		
	return result;
}

template<typename T>
matrix<T> matrix<T>::minimum(const T& value)
{
	matrix<T> result(rows, cols, 0);
	for (unsigned i = 0; i < rows; i++)
	{
		for (unsigned j = 0; j < cols; j++)
		{
			result(i, j) = std::min(mat[i][j], value);
		}
	}

	return result;
}

// element wise max
template<typename T>
matrix<T> matrix<T>::maximum_mat(matrix<T>& rhs)
{
	if (cols != rhs.get_cols() || rows != rhs.get_rows())
	{
		std::cout << "Matricies don't align: " << rows << ", " << cols << " max " << rhs.get_rows() << ", " << rhs.get_cols() << std::endl;
	}
	matrix<T> result(rows, cols, 0);
	for (unsigned i = 0; i < rows; i++)
	{
		for (unsigned j = 0; j < cols; j++)
		{
			result(i, j) = std::max(mat[i][j], rhs[i][j]);
		}
	}

	return result;
}
// element wise min
template<typename T>
matrix<T> matrix<T>::minimum_mat(matrix<T>& rhs)
{
	if (cols != rhs.get_cols() || rows != rhs.get_rows())
	{
		std::cout << "Matricies don't align: " << rows << ", " << cols << " min " << rhs.get_rows() << ", " << rhs.get_cols() << std::endl;
	}
	matrix<T> result(rows, cols, 0);
	for (unsigned i = 0; i < rows; i++)
	{
		for (unsigned j = 0; j < cols; j++)
		{
			result(i, j) = std::min(mat[i][j], rhs[i][j]);
		}
	}

	return result;
}

template<typename T>
matrix<T> matrix<T>::exponent()
{
	matrix<T> result(rows, cols, 0);
	for (unsigned i = 0; i < rows; i++)
	{
		for (unsigned j = 0; j < cols; j++)
		{
			result(i, j) = std::exp(mat[i][j]);
		}
	}

	return result;
}

template<typename T>
std::string matrix<T>::get_size_str()
{
	int r = get_rows();
	int c = get_cols();
	std::ostringstream oss;
	oss << "Rows: " << r << ", Cols: " << c;
	std::string result = oss.str();
	return result;
}

// Access the individual elements                                                                                                                                             
template<typename T>
T& matrix<T>::operator()(const unsigned& row, const unsigned& col)
{
	return this->mat[row][col];
}

// Access the individual elements (const)                                                                                                                                     
template<typename T>
const T& matrix<T>::operator()(const unsigned& row, const unsigned& col) const
{
	return this->mat[row][col];
}

// Access the individual elements                                                                                                                                             
template<typename T>
std::vector<T>& matrix<T>::operator()(const unsigned& row)
{
	return this->mat[row];
}

// Access the individual elements (const)                                                                                                                                     
template<typename T>
const std::vector<T>& matrix<T>::operator()(const unsigned& row) const
{
	return this->mat[row];
}

// Get the number of rows of the matrix                                                                                                                                       
template<typename T>
unsigned matrix<T>::get_rows() const 
{
	return this->rows;
}

// Get the number of columns of the matrix                                                                                                                                    
template<typename T>
unsigned matrix<T>::get_cols() const 
{
	return this->cols;
}

#endif