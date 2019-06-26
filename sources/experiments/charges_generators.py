from sources.pdesolver.finite_differences_method.charge_distribution import ChargeDistribution


def make_central_charge(geometry):
    charges = ChargeDistribution(geometry)
    charges.add(len(geometry.X)//2, len(geometry.Y)//2, -10.0)
    return charges

def make_single_charge(geometry, x, y, value):
    charges = ChargeDistribution(geometry)
    charges.add((int)(len(geometry.X) * x), (int)(len(geometry.Y) * y), value)
    return charges

def make_double_charge(geometry, x1, y1, x2, y2, value):
    charges = ChargeDistribution(geometry)
    charges.add((int)(len(geometry.X) * x1), (int)(len(geometry.Y) * y1), value)
    charges.add((int)(len(geometry.X) * x2), (int)(len(geometry.Y) * y2), value)
    return charges

def make_n_fold_charge(geometry, x, y, index, n, value, variateSign=False):
    charges = ChargeDistribution(geometry)
    for i in range(n):
        charges.add((int)(len(geometry.X) * x[index*n+i]), (int)(len(geometry.Y) * y[index*n+i]), value)
        if variateSign:
            value *= -1

    return charges

def make_n_fold_charge_from_list(geometry, charges_list, value, variateSign=False):
    charges = ChargeDistribution(geometry)
    for charge_tuple in charges_list:
        column = charge_tuple[0]
        row = charge_tuple[1]
        charges.add((int)(len(geometry.X) * column), (int)(len(geometry.Y) * row), value)
        if variateSign:
            value *= -1

    return charges


def make_dipol(g):
    charges = ChargeDistribution(g)
    charges.add(8, 16, 200.0)
    charges.add(-8, 16, -200.0)
    return charges


def make_quadrupol(g):
    charges = ChargeDistribution(g)
    charge = -1000.0
    charges.add((int)(len(g.X)/4), (int)(len(g.X)/4), charge)
    charges.add((int)(len(g.X)/4), (int)(len(g.X)/4*3), -charge)
    charges.add((int)(len(g.X)/4*3), (int)(len(g.X)/4), -charge)
    charges.add((int)(len(g.X)/4*3), (int)(len(g.X)/4*3), charge)
    return charges


def make_comb(g, delta):
    charges = ChargeDistribution(g)
    startX = g.rect.x1 + 8.0
    stopX = 24.0
    charge = -10.0
    for i in range((int)((stopX - startX) / delta + 1)):
        charges.add(startX + i * delta, 20.0, charge)

    return charges