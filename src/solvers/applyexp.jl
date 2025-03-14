using LinearAlgebra: dot, norm
using Printf: @printf

# This helper remains unchanged. It linearly combines the basis vectors
# with coefficients obtained from the exponentiated Hessenberg matrix.
function assemble_lanczos_vecs(basis_vectors, linear_comb, norm)
  xt = norm * linear_comb[1] * basis_vectors[1]
  for i in 2:length(basis_vectors)
    xt += norm * linear_comb[i] * basis_vectors[i]
  end
  return xt
end

struct ApplyExpInfo
  numops::Int
  converged::Int
end

"""
    applyexp(H, tau::Number, x0; maxiter=30, tol=1e-12, outputlevel=0, normcutoff=1e-7)

This function approximates the action of the exponential operator
\[
e^{\tau H}x_0,
\]
using an Arnoldi iteration to build a Krylov subspace. The method is now
suitable for non-Hermitian operators (since it uses full orthogonalization)
and returns the evolved state along with an `ApplyExpInfo` structure that
records the number of matrix-vector multiplications and whether convergence
was achieved.

The interface and return values are kept identical to the Lanczos version.
"""
function applyexp(H, tau::Number, x0; maxiter=30, tol=1e-12, outputlevel=0, normcutoff=1e-7)
  # Initialize Arnoldi basis with normalized starting vector.
  v1 = copy(x0)
  nrm = norm(v1)
  v1 /= nrm
  arnoldi_vectors = [v1]

  # Allocate a matrix to hold the Hessenberg representation.
  ElT = promote_type(typeof(tau), eltype(x0))
  bigH = zeros(ElT, maxiter + 3, maxiter + 3)

  nmatvec = 0

  for iter in 1:maxiter
    # Matrix-vector multiplication on the current basis vector.
    w = H(arnoldi_vectors[iter])
    nmatvec += 1

    # Full orthogonalization against all previously computed basis vectors.
    for j in 1:iter
      h = dot(w, arnoldi_vectors[j])
      bigH[j, iter] = h
      w -= h * arnoldi_vectors[j]
    end

    h_next = norm(w)
    bigH[iter + 1, iter] = h_next

    # Check for breakdown (if the new vector is nearly zero).
    if abs(h_next) < normcutoff
      tmat_size = iter + 1
      tmat = bigH[1:tmat_size, 1:tmat_size]
      tmat_exp = exp(tau * tmat)
      linear_comb = tmat_exp[:, 1]
      xt = assemble_lanczos_vecs(arnoldi_vectors, linear_comb, nrm)
      return xt, ApplyExpInfo(nmatvec, 1)
    end

    # Append the new Arnoldi vector.
    push!(arnoldi_vectors, w / h_next)

    # --- Convergence check using an extended Hessenberg matrix ---
    # We mimic the Lanczos code by defining an extended matrix.
    tmat_size = iter + 1  # number of basis vectors so far
    tmat_ext_size = tmat_size + 2
    tmat_ext = bigH[1:tmat_ext_size, 1:tmat_ext_size]
    # Adjust entries as in the Lanczos version.
    tmat_ext[tmat_size - 1, tmat_size] = 0.0
    tmat_ext[tmat_size + 1, tmat_size] = 1.0

    tmat_ext_exp = exp(tau * tmat_ext)

    ϕ1 = abs(nrm * tmat_ext_exp[tmat_size, 1])
    ϕ2 = abs(nrm * tmat_ext_exp[tmat_size + 1, 1] * h_next)
    if ϕ1 > 10 * ϕ2
      err_est = ϕ2
    elseif ϕ1 > ϕ2
      err_est = (ϕ1 * ϕ2) / (ϕ1 - ϕ2)
    else
      err_est = ϕ1
    end

    if outputlevel >= 3
      @printf("  Iteration: %d, Error: %.2E\n", iter, err_est)
    end

    if (err_est < tol) || (iter == maxiter)
      converged = 1
      if iter == maxiter
        println("warning: applyexp not converged in $maxiter steps")
        converged = 0
      end
      # Use the extended matrix coefficients to form the final state.
      linear_comb = tmat_ext_exp[:, 1]
      xt = assemble_lanczos_vecs(arnoldi_vectors, linear_comb, nrm)
      if outputlevel >= 3
        println("  Number of iterations: $iter")
      end
      return xt, ApplyExpInfo(nmatvec, converged)
    end
  end

  println("In applyexp, number of matrix-vector multiplies: ", nmatvec)
  return x0
end
