using NumericalDifferentiation, PyPlot

x = range(-1, stop=1, length=101)
y = abs.(x) + (rand(length(x)) .- .5)/5

u_TV = differentiate(x, y, TotalVariation(), 0.2, 1e-3, maxit=200)
u_TK = differentiate(x, y, Tikhonov(), 0.0002)

figure()
plot(x, y .- y[1],
     x, integrationoperator(x,u_TV),
     x, integrationoperator(x,u_TK))
plot(x, u_TV,
     x, u_TK)

z = sin.(π*x) + (rand(length(x)) .- .5)/5

u_TV = differentiate(x, z, TotalVariation(), 0.2, 1e-3, maxit=200)
u_TK = differentiate(x, z, Tikhonov(), 0.0002)

figure()
plot(x, z .- z[1],
     x, integrationoperator(x,u_TV),
     x, integrationoperator(x,u_TK))

plot(x, u_TV,
     x, u_TK)
