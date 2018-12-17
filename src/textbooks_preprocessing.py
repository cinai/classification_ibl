# -*- coding: utf-8 -*-
'''
Module with methods to extract content from school textbooks.
'''

__author__ = "Abelino Jiménez"

import PyPDF2, glob, os
from docx import Document

'''
PDF methods
'''

'''
extract text for eache page in a pdf file

str -> [str]
input:
  file: path of the pdf file
output:
  pages: list of text for each page in the pdf file
'''
def extract_pages(file):
    print('\nExtracting file :' + file)
    pdfFileObj = open(file,'rb')
    try:
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj,strict=False)
    except:
        pdfFileObj.close()
        return []
    N = pdfReader.numPages
    pages = []
    for i in range(N):
        print('%d/%d'%(i+1,N),end = '\r')
        try:
            pageObj = pdfReader.getPage(i)
            page = pageObj.extractText()
            pages.append(page)
        except:
            print("\tPage %d coundn't be extracted..."%(i))
    return pages

'''
save text of pages into text files

str,[],str,int -> None
input:
  file: name of the original file
  pages: list of text to save
  destination: path of destination
'''
def save_pages(file, pages, destination):
    print('\nPrinting file :' + file)
    name = os.path.basename(file)[:-4]
    N = len(pages)
    if N>0:
        if not(os.path.isdir(destination)):
            os.mkdir(destination)
        for i in range(N):
            print('%d/%d'%(i+1,N),end = '\r')
            file_result = destination + '/' + name + '_page_' + str(i+1) +'.txt'
            f = open(file_result,"w",encoding='utf8')
            f.write(pages[i])
            f.close()

'''
Word Methods
'''

'''
extract paragraphs from a word document

str -> [str]
input:
  file: path of the word file
output:
  paragraphs: list of paragraphs of the word file
'''
def extract_paragraphs(file):
    a_doc = Document(file)    
    paragraphs = []
    for s in a_doc.paragraphs:
        words = s.text.split()
        if len(words)!=0:
            paragraphs.append(s.text)
    return paragraphs

'''
save paragraphs into text files

str,[],str,int -> None
input:
  textbook: name of the original file
  paragraphs: list of paragraphs to save
  destination: path of destination
  n_paragraphs: number of paragraphs to be grouped per saved file
'''
def save_paragraphs(textbook,paragraphs,destination,n_paragraphs=10):
    textbook = os.path.basename(textbook)[:-5] # remove word extension
    print('\nPrinting file :' + textbook)
    N = len(paragraphs)
    if not os.path.exists(destination):
        os.makedirs(destination)
    for p in range(0,len(paragraphs),n_paragraphs):
        print('%d/%d'%(p+1,N),end = '\r')
        count = int(p/n_paragraphs)
        file_name = textbook[:-5]+'_{}.txt'.format(count)
        file_path = os.path.join(destination,file_name)
        with open(file_path,'w',encoding='utf-8') as f:
            for i in range(p,p+n_paragraphs):
                if i < len(paragraphs):
                    f.write(paragraphs[i])
