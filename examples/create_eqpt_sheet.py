#!/usr/bin/env python3
# TelecomInfraProject/gnpy/examples
# Module name : create_eqpt_sheet.py
# Version : 
# License : BSD 3-Clause Licence
# Copyright (c) 2018, Telecom Infra Project

"""
@author: esther.lerouzic
@author: jeanluc-auge
xls parser, that can be called to create the mandatory 'City' column in Eqpt sheet. 
If not present in sheet Nodes, the  'Type' column is implicitely determined based on the 
topology: degree 2 = ILA, other degrees = ROADM. The value is also corrected to ROADM if the user 
specifies an ILA of degree != 2.
 

"""
from sys import exit
try:
    from xlrd import open_workbook
except ModuleNotFoundError:
    exit('Required: `pip install xlrd`')
from argparse import ArgumentParser
from collections import namedtuple, defaultdict


class Shortlink(namedtuple('Link', 'src dest')):
    def __new__(cls, src,dest):
        src = src 
        dest = dest 
        return super().__new__(cls,src,dest)

class Shortnode(namedtuple('Node', 'nodename eqt')):
    def __new__(cls, nodename,eqt):
        nodename = nodename 
        eqt = eqt
        return super().__new__(cls,nodename,eqt)

parser = ArgumentParser()
parser.add_argument('workbook', nargs='?', default='meshTopologyExampleV2.xls',
    help = 'create the mandatory columns in Eqpt sheet  ')
all_rows = lambda sh, start=0: (sh.row(x) for x in range(start, sh.nrows))

def read_excel(input_filename):
    with open_workbook(input_filename) as wb:
        # reading Links sheet
        links_sheet = wb.sheet_by_name('Links')
        links = []
        nodeoccuranceinlinks = []
        links_by_src = defaultdict(list)
        links_by_dest = defaultdict(list)
        for row in all_rows(links_sheet, start=5):
            links.append(Shortlink(row[0].value,row[1].value))
            links_by_src[row[0].value].append(Shortnode(row[1].value,''))
            links_by_dest[row[1].value].append(Shortnode(row[0].value,''))
            #print(f'source {links[len(links)-1].src} dest {links[len(links)-1].dest}')
            nodeoccuranceinlinks.append(row[0].value)
            nodeoccuranceinlinks.append(row[1].value)

        # reading Nodes sheet
        nodes_sheet = wb.sheet_by_name('Nodes')
        nodes = []
        node_degree = []
        for row in all_rows(nodes_sheet, start=5) :
            
            temp_eqt = row[6].value
            # verify node degree to confirm eqt type
            node_degree.append(nodeoccuranceinlinks.count(row[0].value))
            if temp_eqt.lower() == 'ila' and nodeoccuranceinlinks.count(row[0].value) !=2 :
                print(f'Inconsistancy: node {nodes[len(nodes)-1]} has degree \
                    {node_degree[len(nodes)-1]} and can not be an ILA ... replaced by ROADM')
                temp_eqt = 'ROADM'
            if temp_eqt == '' and nodeoccuranceinlinks.count(row[0].value) == 2 :
                temp_eqt = 'ILA'
            if temp_eqt == '' and nodeoccuranceinlinks.count(row[0].value) != 2 :
                temp_eqt = 'ROADM'
            # print(f'node {nodes[len(nodes)-1]} eqt {temp_eqt}') 
            nodes.append(Shortnode(row[0].value,temp_eqt))
            # print(len(nodes)-1)
            print(f'reading: node {nodes[len(nodes)-1].nodename} eqt {temp_eqt}')        
        return links,nodes, links_by_src , links_by_dest

def create_eqt_template(links,nodes, links_by_src , links_by_dest, input_filename):
    output_filename = f'{input_filename[:-4]}_eqt_sheet.txt'
    with open(output_filename,'w') as my_file :
        # print header similar to excel
        my_file.write('OPTIONAL\n\n\n\
           \t\tNode a egress amp (from a to z)\t\t\t\t\tNode a ingress amp (from z to a) \
           \nNode A \tNode Z \tamp type \tatt_in \tamp gain \ttilt \tatt_out\
           amp type   \tatt_in \tamp gain   \ttilt   \tatt_out\n')                                            

        tab = []
        temp = []
        i = 0
        for lk in links:
            temp = [lk.src , lk.dest]
            tab.append(temp)
            # print(temp)
            my_file.write(f'{temp[0]}\t{temp[1]}\n')
        for n in nodes :
            if n.eqt.lower() == 'roadm' :
                for src in  links_by_dest[n.nodename] :
                    temp = [n.nodename , src.nodename]
                    tab.append(temp)
                    # print(temp)
                    my_file.write(f'{temp[0]}\t{temp[1]}\n')
            i = i + 1
        print(f'File {output_filename} successfully created with Node A - Node Z ' +
        ' entries for Eqpt sheet in excel file.')

if __name__ == '__main__':
    args = parser.parse_args()
    input_filename = args.workbook
    links,nodes,links_by_src, links_by_dest = read_excel(input_filename)
    create_eqt_template(links,nodes, links_by_src , links_by_dest , input_filename)


