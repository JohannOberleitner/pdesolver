from sources.pdesolver.finite_differences_method.charge_distribution import ChargeDistribution


def make_central_charge(geometry):
    charges = ChargeDistribution(geometry)
    charges.add(len(geometry.X)//2, len(geometry.Y)//2, -10.0)
    return charges


def make_dipol(g):
    charges = ChargeDistribution(g)
    charges.add(8, 16, 200.0)
    charges.add(-8, 16, -200.0)
    return charges


def make_quadrupol(g):
    charges = ChargeDistribution(g)
    charge = -1000.0
    charges.add(8, 8, charge)
    charges.add(8, 24, -charge)
    charges.add(24, 8, -charge)
    charges.add(24, 24, charge)
    return charges


def make_comb(g, delta):
    charges = ChargeDistribution(g)
    startX = g.rect.x1 + 8.0
    stopX = 24.0
    charge = -10.0
    for i in range((int)((stopX - startX) / delta + 1)):
        charges.add(startX + i * delta, 20.0, charge)

    return charges