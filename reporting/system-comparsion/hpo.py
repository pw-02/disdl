import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec

#figure data
tensorsocket_label = "TensorSocket"
disdl_label = "DisDL"
coordl_label = "ReCoorDL"
line_width = 2
font_size = 12

visual_map_plot = {
    tensorsocket_label: {'color': '#FDA300', 'linestyle': '-', 'linewidth': line_width, 'edgecolor': 'black', 'hatch': '...','alpha': 1.0}, #indianred
    disdl_label: {'color': 'black', 'linestyle': '-', 'linewidth': line_width,'edgecolor': 'black', 'hatch':'', 'alpha': 1.0}, #steelblue
    coordl_label: {'color': '#A3B6CE', 'linestyle': '-', 'linewidth': line_width, 'edgecolor': 'black', 'hatch': '','alpha': 1.0}, #indianred

}

visual_map_plot = {
    coordl_label: {'color': '#A3B6CE', 'linestyle': '-', 'linewidth': line_width, 'edgecolor': 'black', 'hatch': '','alpha': 1.0}, #indianred
    disdl_label: {'color': 'black', 'linestyle': '-', 'linewidth': line_width,'edgecolor': 'black', 'hatch':'', 'alpha': 1.0}, #steelblue
    tensorsocket_label: {'color': '#FDA300', 'linestyle': '-', 'linewidth': line_width, 'edgecolor': 'black', 'hatch': '','alpha': 1.0}, #indianred
}


categories = [
    'ViT-B/32', 'LeViT-128', 'ResNet-50', 'ResNet-18', 'RegNetX4GF', 'MobileNetV3L', 'ALBEF'
]

tenorsocket_values = [1006.741940344406, 990.935084942362, 1336.221136,1875.009251,  1954.195471, 2015.392322,945.201957825627] 
coordl_values = [996.741940344406, 996.935084942362, 1356.221136,1835.009251,  1966.195471, 1998.392322,922.201957825627]  
disdl_values = [1537.38446459459,2565.0288440519,1452.79088984116,2837.91602839437,1983.80185065615,2863.27618270038,990.485820126382]  

#DIVIDE values bt 128 to get batches per second
tenorsocket_values = [x/1 for x in tenorsocket_values]
coordl_values = [x/1 for x in coordl_values]

disdl_values = [x/1 for x in disdl_values]

fig = plt.figure(figsize=(12, 4))
gs = gridspec.GridSpec(1, 1, width_ratios=[1])
ax1 = fig.add_subplot(gs[0, 0])

# Set up the x locations for the bars
x = np.arange(len(categories))
width = 0.3  # Adjusted width to accommodate three bars

ax1.bar(x - width, tenorsocket_values, 
        width, 
        label=tensorsocket_label, 
        color=visual_map_plot[tensorsocket_label]['color'], 
          edgecolor = 'black', alpha=visual_map_plot[tensorsocket_label]['alpha'],
          hatch=visual_map_plot[tensorsocket_label]['hatch'])
ax1.bar(x, coordl_values, width, label=coordl_label, 
        color=visual_map_plot[coordl_label]['color'], edgecolor='black',
        alpha=visual_map_plot[coordl_label]['alpha'], hatch=visual_map_plot[coordl_label]['hatch'])
ax1.bar(x + width, disdl_values, width, label=disdl_label, color=visual_map_plot[disdl_label]['color'], edgecolor = 'black',
          alpha=visual_map_plot[disdl_label]['alpha'],
          hatch=visual_map_plot[disdl_label]['hatch'])

ax1.set_ylabel("Aggregated samples/s", fontsize=12)
ax1.set_xticks(x - width / 2)
ax1.set_xticklabels(categories)
ax1.legend(loc='upper center', fontsize=font_size, ncol=3, frameon=True)
ax1.set_ylim(0, 3500)
#set font size for all lables and ticks
ax1.tick_params(axis='both', which='major', labelsize=font_size)
ax1.tick_params(axis='both', which='minor', labelsize=font_size)
ax1.xaxis.label.set_size(font_size)
ax1.yaxis.label.set_size(font_size)
ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
plt.tight_layout()
plt.show()
