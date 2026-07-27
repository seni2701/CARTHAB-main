import os
import arcpy
from arcpy import AddMessage
import geopandas as gpd
import pandas as pd
from neptune.quality import *
from bs4 import BeautifulSoup
from statsmodels.stats.descriptivestats import describe
import platform

pd.set_option('display.float_format', '{:.2f}'.format)

river_code = arcpy.GetParameterAsText(0)
transects_n1_path = arcpy.GetParameterAsText(1)
transects_lb_path = arcpy.GetParameterAsText(2)
index_ortho_path = arcpy.GetParameterAsText(3)
c_value_path = arcpy.GetParameterAsText(4)
transects_n2_path = arcpy.GetParameterAsText(5)
transects_n2_trib_path = arcpy.GetParameterAsText(6)
out_html_path = arcpy.GetParameterAsText(7)

soup = BeautifulSoup('<html></html>', 'html.parser')

style_tag = soup.new_tag("style")
style_tag.string = """
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f4f7f6;
    margin: 0;
    padding: 20px;
}

.container {
    max-width: 1000px;
    margin: 0 auto;
    background: #fff;
    padding: 30px;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

h1 {
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
    margin-top: 40px;
}

h1:first-of-type {
    margin-top: 0;
}

h2 {
    font-size: 1.2em;
    margin-top: 20px;
}

.warning { color: #d35400; background-color: #fef5e7; padding: 10px; border-left: 5px solid #e67e22; border-radius: 4px; }
.error { color: #c0392b; background-color: #f9ebea; padding: 10px; border-left: 5px solid #e74c3c; border-radius: 4px; }
.errortitle { color: #c0392b; margin-top: 2em; font-weight: bold; }
.nowarning { color: #27ae60; background-color: #eafaf1; padding: 10px; border-left: 5px solid #2ecc71; border-radius: 4px; }
.noerror { color: #27ae60; background-color: #eafaf1; padding: 10px; border-left: 5px solid #2ecc71; border-radius: 4px; }

.table_component {
    margin: 20px 0;
    overflow-x: auto;
}

.table_component table {
    border-collapse: collapse;
    width: 100%;
    background-color: white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.table_component th {
    background-color: #3498db;
    color: white;
    font-weight: 600;
    text-align: left;
    padding: 12px 15px;
}

.table_component td {
    padding: 10px 15px;
    border-bottom: 1px solid #ddd;
}

.table_component tr:hover {
    background-color: #f1f1f1;
}

.table_component tr:nth-child(even) {
    background-color: #f9f9f9;
}
"""
soup.html.append(style_tag)

body_tag = soup.new_tag("body")
soup.html.append(body_tag)
container_div = soup.new_tag("div", **{'class': 'container'})
body_tag.append(container_div)

h2_tag = soup.new_tag('h2', **{'class': 'warning', 'style': 'margin-bottom: 30px;'})
h2_tag.string = ("Les avertissements ne bloquent en rien la progression pour les prochaines étapes et ne "
                 "nécessitent pas de modifications obligatoires")
container_div.append(h2_tag)

stats = ["nobs", "missing", "mean", "median", "max", "min", 'percentiles']
percentiles = [1, 10, 25, 50, 75, 99]

no_errors_warnings = True
index_ortho_opened = False

if transects_n1_path:
    h1_tag = soup.new_tag('h1')
    h1_tag.string = 'Couche transects N1 :'
    container_div.append(h1_tag)

    transects_n1 = gpd.read_file(transects_n1_path)
    warnings, errors = check_transects_n1(transects_n1, river_code)

    mask = (transects_n1['Backwater'] == 0)
    descriptive_stats = describe(transects_n1.loc[mask, ['FACC_COR', 'Q2_spli', 'WAT_WIDTH', 'MAX_WIDTH']],
                                 stats=stats, percentiles=percentiles)

    missing_values = check_desc_stats(descriptive_stats, 0.10)
    if missing_values:
        errors.extend(missing_values)

    if index_ortho_path:
        index = gpd.read_file(index_ortho_path)
        index_geom = index.unary_union
        index_ortho_opened = True

        if any(~transects_n1.intersects(index_geom)):
            errors.append(f"Certains transects du fichier Transects N1 n'intersects pas le fichier ortho d'index")

    if warnings:
        no_errors_warnings = False
        warning_title = 'Il y a des avertissements :'
        h2_tag = soup.new_tag('h2', **{'class': 'warning'})
        h2_tag.string = warning_title
        container_div.append(h2_tag)

        for warning in warnings:
            warning_item = soup.new_tag('p', **{'class': 'warning'})
            warning_item.string = warning
            container_div.append(warning_item)
    else:
        p_tag = soup.new_tag('p', **{'class': 'nowarning'})
        p_tag.string = "Il n'y a pas d'avertissement"
        container_div.append(p_tag)

    if errors:
        no_errors_warnings = False
        error_title = "Il y a des erreurs empêchant le prochain processus d'être accompli :"
        h2_tag = soup.new_tag('h2', **{'class': 'errortitle'})
        h2_tag.string = error_title
        container_div.append(h2_tag)

        for error in errors:
            error_item = soup.new_tag('p', **{'class': 'error'})
            error_item.string = error
            container_div.append(error_item)
    else:
        p_tag = soup.new_tag('p', **{'class': 'noerror'})
        p_tag.string = "Il n'y a pas d'erreur"
        container_div.append(p_tag)

    table_wrapper_div = soup.new_tag('div', **{'class': 'table_component'})
    html_table = descriptive_stats.to_html()
    table_tag = BeautifulSoup(html_table, 'html.parser')
    table_wrapper_div.append(table_tag)
    container_div.append(table_wrapper_div)

if transects_lb_path:
    h1_tag = soup.new_tag('h1')
    h1_tag.string = 'Couche transects lidar brut :'
    container_div.append(h1_tag)

    transects_lb = gpd.read_file(transects_lb_path)
    warnings, errors = check_transects_lb(transects_lb, river_code)

    mask = (transects_lb['Backwater'] == 0)
    descriptive_stats = describe(transects_lb.loc[mask, ['LB_Q25']], stats=stats, percentiles=percentiles)

    missing_values = check_desc_stats(descriptive_stats, 0.10)
    if missing_values:
        errors.extend(missing_values)

    if warnings:
        no_errors_warnings = False
        warning_title = 'Il y a des avertissements :'
        h2_tag = soup.new_tag('h2', **{'class': 'warning'})
        h2_tag.string = warning_title
        container_div.append(h2_tag)

        for warning in warnings:
            warning_item = soup.new_tag('p', **{'class': 'warning'})
            warning_item.string = warning
            container_div.append(warning_item)
    else:
        p_tag = soup.new_tag('p', **{'class': 'nowarning'})
        p_tag.string = "\nIl n'y a pas d'avertissement"
        container_div.append(p_tag)

    if errors:
        no_errors_warnings = False
        error_title = "Il y a des erreurs empêchant le prochain processus d'être accompli :"
        h2_tag = soup.new_tag('h2', **{'class': 'errortitle'})
        h2_tag.string = error_title
        container_div.append(h2_tag)

        for error in errors:
            error_item = soup.new_tag('p', **{'class': 'error'})
            error_item.string = error
            container_div.append(error_item)
    else:
        p_tag = soup.new_tag('p', **{'class': 'noerror'})
        p_tag.string = "Il n'y a pas d'erreur"
        container_div.append(p_tag)

    table_wrapper_div = soup.new_tag('div', **{'class': 'table_component'})
    mask = (transects_lb['Backwater'] == 0)
    html_table = descriptive_stats.to_html()
    table_tag = BeautifulSoup(html_table, 'html.parser')
    table_wrapper_div.append(table_tag)
    container_div.append(table_wrapper_div)

if index_ortho_path:
    h1_tag = soup.new_tag('h1')
    h1_tag.string = 'Couche index ortho :'
    container_div.append(h1_tag)
    if not index_ortho_opened:
        index = gpd.read_file(index_ortho_path)
    warnings, errors = check_ortho_index(index, river_code)

    if warnings:
        no_errors_warnings = False
        warning_title = 'Il y a des avertissements :'
        h2_tag = soup.new_tag('h2', **{'class': 'warning'})
        h2_tag.string = warning_title
        container_div.append(h2_tag)

        for warning in warnings:
            warning_item = soup.new_tag('p', **{'class': 'warning'})
            warning_item.string = warning
            container_div.append(warning_item)
    else:
        p_tag = soup.new_tag('p', **{'class': 'nowarning'})
        p_tag.string = "Il n'y a pas d'avertissement"
        container_div.append(p_tag)

    if errors:
        no_errors_warnings = False
        error_title = "Il y a des erreurs empêchant le prochain processus d'être accompli :"
        h2_tag = soup.new_tag('h2', **{'class': 'errortitle'})
        h2_tag.string = error_title
        container_div.append(h2_tag)

        for error in errors:
            error_item = soup.new_tag('p', **{'class': 'error'})
            error_item.string = error
            container_div.append(error_item)
    else:
        p_tag = soup.new_tag('p', **{'class': 'noerror'})
        p_tag.string = "\nIl n'y a pas d'erreur"
        container_div.append(p_tag)

if c_value_path:
    h1_tag = soup.new_tag('h1')
    h1_tag.string = 'Table des valeurs de c :'
    container_div.append(h1_tag)

    xls_file = pd.ExcelFile(c_value_path)
    if 'valeurs_c' not in xls_file.sheet_names:
        no_errors_warnings = False
        error_title = ("Il y a des erreurs dans le tableau des valeurs de c :\nLa table 'valeurs_c' n'est pas dans "
                       "les onglets du fichier fourni")
        h2_tag = soup.new_tag('h2', **{'class': 'errortitle'})
        h2_tag.string = error_title
        container_div.append(h2_tag)

    else:
        c_value_sheet = pd.read_excel(c_value_path, sheet_name='valeurs_c')
        warnings, errors = check_c_value(c_value_sheet)

        if warnings:
            no_errors_warnings = False
            warning_title = 'Il y a des avertissements dans :'
            h2_tag = soup.new_tag('h2', **{'class': 'warning'})
            h2_tag.string = warning_title
            container_div.append(h2_tag)

            for warning in warnings:
                warning_item = soup.new_tag('p', **{'class': 'warning'})
                warning_item.string = warning
                container_div.append(warning_item)
        else:
            p_tag = soup.new_tag('p', **{'class': 'nowarning'})
            p_tag.string = "Il n'y a pas d'avertissement"
            container_div.append(p_tag)

        if errors:
            no_errors_warnings = False
            error_title = "Il y a des erreurs enpéchant le prochain processus d'être accomplie :"
            h2_tag = soup.new_tag('h2', **{'class': 'errortitle'})
            h2_tag.string = error_title
            container_div.append(h2_tag)

            for error in errors:
                error_item = soup.new_tag('p', **{'class': 'error'})
                error_item.string = error
                container_div.append(error_item)
        else:
            p_tag = soup.new_tag('p', **{'class': 'noerror'})
            p_tag.string = "Il n'y a pas d'erreur"
            container_div.append(p_tag)

if transects_n2_path:
    h1_tag = soup.new_tag('h1')
    h1_tag.string = 'Couche transects N2 :'
    container_div.append(h1_tag)

    transects_n2 = gpd.read_file(transects_n2_path)
    warnings, errors = check_transects_n2(transects_n2, river_code)

    mask = (transects_n2['Backwater'] == 0)
    descriptive_stats = describe(transects_n2.loc[mask, ['Q_IMG_spli', 'Pente', 'D84',  'Surface', 'Backwater', 'Lac']],
                                 stats=stats, percentiles=percentiles)

    missing_values = check_desc_stats(descriptive_stats, 0.10)
    if missing_values:
        errors.extend(missing_values)

    if warnings:
        no_errors_warnings = False
        warning_title = 'Il y a des avertissements :'
        h2_tag = soup.new_tag('h2', **{'class': 'warning'})
        h2_tag.string = warning_title
        container_div.append(h2_tag)

        for warning in warnings:
            warning_item = soup.new_tag('p', **{'class': 'warning'})
            warning_item.string = warning
            container_div.append(warning_item)
    else:
        p_tag = soup.new_tag('p', **{'class': 'nowarning'})
        p_tag.string = "Il n'y a pas d'avertissement"
        container_div.append(p_tag)

    if errors:
        no_errors_warnings = False
        error_title = "Il y a des erreurs empêchant le prochain processus d'être accompli :"
        h2_tag = soup.new_tag('h2', **{'class': 'errortitle'})
        h2_tag.string = error_title
        container_div.append(h2_tag)

        for error in errors:
            error_item = soup.new_tag('p', **{'class': 'error'})
            error_item.string = error
            container_div.append(error_item)
    else:
        p_tag = soup.new_tag('p', **{'class': 'noerror'})
        p_tag.string = "Il n'y a pas d'erreur"
        container_div.append(p_tag)

    table_wrapper_div = soup.new_tag('div', **{'class': 'table_component'})
    html_table = descriptive_stats.to_html()
    table_tag = BeautifulSoup(html_table, 'html.parser')
    table_wrapper_div.append(table_tag)
    container_div.append(table_wrapper_div)


if transects_n2_trib_path:
    h1_tag = soup.new_tag('h1')
    h1_tag.string = 'Couche transects N2 tributaires :'
    container_div.append(h1_tag)

    transects_n2_trib = gpd.read_file(transects_n2_trib_path)
    warnings, errors = check_transects_n2_trib(transects_n2_trib, river_code)

    transects_n2_trib['Accessible'] = transects_n2_trib['Accessible'].astype('category')
    stats = stats + ['top', 'freq']
    descriptive_stats = describe(transects_n2_trib[['Q_IMG', 'Pente', 'D84', 'Accessible']],
                                 stats=stats, percentiles=percentiles, ntop=2)

    missing_values = check_desc_stats(descriptive_stats, 0.10)
    if missing_values:
        errors.extend(missing_values)

    if warnings:
        no_errors_warnings = False
        warning_title = 'Il y a des avertissements :'
        h2_tag = soup.new_tag('h2', **{'class': 'warning'})
        h2_tag.string = warning_title
        container_div.append(h2_tag)

        for warning in warnings:
            warning_item = soup.new_tag('p', **{'class': 'warning'})
            warning_item.string = warning
            container_div.append(warning_item)
    else:
        p_tag = soup.new_tag('p', **{'class': 'nowarning'})
        p_tag.string = "Il n'y a pas d'avertissement"
        container_div.append(p_tag)

    if errors:
        no_errors_warnings = False
        error_title = "Il y a des erreurs empêchant le prochain processus d'être accompli :"
        h2_tag = soup.new_tag('h2', **{'class': 'errortitle'})
        h2_tag.string = error_title
        container_div.append(h2_tag)

        for error in errors:
            error_item = soup.new_tag('p', **{'class': 'error'})
            error_item.string = error
            container_div.append(error_item)
    else:
        p_tag = soup.new_tag('p', **{'class': 'noerror'})
        p_tag.string = "Il n'y a pas d'erreur"
        container_div.append(p_tag)

    table_wrapper_div = soup.new_tag('div', **{'class': 'table_component'})
    html_table = descriptive_stats.to_html()
    table_tag = BeautifulSoup(html_table, 'html.parser')
    table_wrapper_div.append(table_tag)
    container_div.append(table_wrapper_div)


if no_errors_warnings:
    h2_tag = container_div.find('h2', {'class': 'warning'},
                       string="Les avertissements ne bloquent en rien la progression pour les prochaines étapes et ne "
                              "nécessitent pas de modifications obligatoires")
    if h2_tag:
        h2_tag.decompose()

with open(out_html_path, 'w') as f:
    f.write(str(soup))

if platform.system() == 'Windows':
    os.startfile(out_html_path)

AddMessage(f"Rapport enregistré sous : {out_html_path}")
