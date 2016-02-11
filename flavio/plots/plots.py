from collections import OrderedDict
import flavio
import matplotlib.pyplot as plt


def error_budget_pie(err_dict, other_cutoff=0.03):
    """Pie chart of an observable's error budget."""
    err_tot = sum(err_dict.values())
    err_dict_sorted = OrderedDict(sorted(err_dict.items(), key=lambda t: -t[1]))
    labels = []
    fracs = []
    small_frac = 0
    for key, value in err_dict_sorted.items():
        frac = value/err_tot
        if frac > other_cutoff:
            labels.append(flavio.Parameter.get_instance(key).tex)
            fracs.append(frac)
        else:
            small_frac += frac
    if small_frac > 0:
        labels.append('other')
        fracs.append(small_frac)
    def my_autopct(pct):
        return r'{p:.1f}\%'.format(p=pct*err_tot)
    plt.axis('equal')
    return plt.pie(fracs, labels=labels, autopct=my_autopct, wedgeprops = {'linewidth':0.5}, colors=flavio.plots.colors.pastel)
