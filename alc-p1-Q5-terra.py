import numpy as np


def resolve_lu(A, b):
    
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(A)

    L = np.eye(n)
    U = np.array(A, dtype=float)

    # 1) Decomposição LU
    for k in range(n):
        # Verifica pivô nulo
        if U[k, k] == 0:
            raise Exception(
                f"Pivô nulo encontrado na posição ({k}, {k}). "
                "A decomposição LU sem pivoteamento não pode ser aplicada. "
                "Utilize uma função alternativa."
            )

        for i in range(k + 1, n):
            # Multiplicador da eliminação -> elemento de L
            m = U[i, k] / U[k, k]
            L[i, k] = m

            # Linha i = linha i - m * linha k
            for j in range(k, n):
                U[i, j] = U[i, j] - m * U[k, j]

            U[i, k] = 0.0

    # 2) Substituição progressiva: Ly = b
    y = np.zeros(n)
    for i in range(n):
        soma = b[i]
        for j in range(i):
            soma = soma - L[i, j] * y[j]
        y[i] = soma / L[i,i]

    # 3) Substituição regressiva: Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = y[i]
        for j in range(i + 1, n):
            soma = soma - U[i, j] * x[j]
        x[i] = soma / U[i, i]

    return L, U, x


# Exemplo de teste
if __name__ == "__main__":
    A = [[2, 1, 1],
         [4, 3, 3],
         [8, 7, 9]]
    b = [4, 10, 24]

    L, U, x = resolve_lu(A, b)

    print("L =\n", L)
    print("U =\n", U)
    print("x =", x)  # esperado: [1. 1. 1.]
