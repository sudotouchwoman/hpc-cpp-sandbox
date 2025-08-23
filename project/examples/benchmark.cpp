#define BOOST_TEST_MODULE blass_dgemm_test_module
#include <boost/test/included/unit_test.hpp>

#include "blas_dgemm.hpp"

BOOST_AUTO_TEST_CASE(naive_sanity_check)
{
	namespace naive = impl::naive;
	constexpr size_t M = 3, K = 3, N = 2;
	constexpr std::array<double, M * K> A_matrix = {
		5, 6, 4, 8, 9, 7, -4, -5, -2,
	};
	constexpr std::array<double, K * N> B_matrix = {
		2, -3, 1, 3, -5, 2,
	};
	constexpr std::array<double, M * N> C_matrix_expect = {
		-18, -20, -15, -33, -37, -27,
	};

	std::array<double, M * N> C_matrix;

	naive::blas_dgemm(M, N, K, A_matrix.data(), B_matrix.data(),
			  C_matrix.data());

	BOOST_CHECK_EQUAL_COLLECTIONS(C_matrix.cbegin(), C_matrix.cend(),
				      C_matrix_expect.cbegin(),
				      C_matrix_expect.cend());
}
