#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:18:49 2019

@author: lejo
"""

import requests
from urllib.request import urlretrieve
import threading
from os import path,remove
import json

DIR = "/tmp"
TOKEN = "" 
_METODOS = ["getUpdates","getMe","sendMessage","getFile"]
MIME_TYPES = ["image/jpeg","image/png"]

last_update = 0

request_format = "https://api.telegram.org/bot{token}/{metodo}"
file_format = "https://api.telegram.org/file/bot{token}/{file_path}"

def load_text(path):
    with open(path) as file:
        text = file.read()
    return text

def InputMediaPhoto(url,caption=None,markdown=False):
    out = {"type":"photo","media":url}
    if caption:
        out["caption"] = caption
        if markdown:
            out["parse_mode"] = "Markdown"
    return out

def PackResponse(captions,urls):
    
    out = [InputMediaPhoto(url,caption,markdown=True)
           for caption,url in zip(captions,urls)]
    return out    
    

def InlineKeyboardButton(text,callback_data):
    button = {"text":text,"callback_data":callback_data}
    return button

def InlineKeyboardMarkup(buttons):
    "array of arrays of InlineKeyboardButton"
    buttons = {"inline_keyboard":buttons}
    return buttons

class LongPoolingUpdate():
  """
  Recibe actualizaciones que pueden contener varios mensajes de diferentes usuarios,
  divide las actualizacion en objetos de la clase "Update" que representan un mensaje unico,
  lleva la cuenta del id del ultimo mensaje al que se ha respondido
  """
  def __init__(self,timeout=5):

    self.last_update_id=0
    self.timeout = timeout
    self._getUpdates = request_format.format(token=TOKEN,metodo="getUpdates")    
      
  def get(self):
    self.UpdatesPars = {"offset":self.last_update_id+1,"timeout":self.timeout}      
    response = requests.get(self._getUpdates,self.UpdatesPars)
    print(response.json())
    resultado = response.json()["result"]#la otra clave es ok:bool
    actualizacion_dividida = [Mensaje(result) for result in resultado 
                              if result.get("message") or 
                                 result.get("callback_query")]
    return actualizacion_dividida
    
class WebhookUpdate():
  def __init__(self,response):
    self.last_update_id = 0
    self.response = response
  def get(self):
    "respuesta: objeto que se recibe por POST requests."
    resultado = self.response
    if isinstance(self.response,dict):
        print("Recibi diccionario")
        return Mensaje(self.response)
    if isinstance(self.response,list):
        print("Recibi lista")
        actualizacion_dividida = [Mensaje(result) for result in resultado]
        return actualizacion_dividida  
    
#response.json().keys()=ok,result

#[result]=list of dicts,keys=update_id,message
        
        
#################################################      
class Mensaje():
    """Representa un mensaje unico, implementa metodos para procesar mensajes con 
    fotos e ignorar aquellos que no tienen"""
    def __init__(self,dictResponse):
        
        self.update_id = dictResponse["update_id"]
        self.mensaje = dictResponse.get("message")
        self.callback_query = dictResponse.get("callback_query")
        self.photo = None
        self.document = None
        
        if self.mensaje:
            self.chat_id = self.mensaje["chat"]["id"]
            self.photo = self.mensaje.get("photo")
            self.document = self.mensaje.get("document")
            self.text = self.mensaje.get("text")
            self.file_id = None
        
        if self.callback_query:
            self.callback_query_id = self.callback_query["id"]
            self.chat_id = self.callback_query["from"]["id"]
            
        if self.hasPhoto():
            self.file_id=self._LargestPhoto()
        
        if self.hasDocImage():
            self.file_id = self.document["file_id"]
        
        
    
    def hasPhoto(self):
        return True if self.photo else False    
    
    def hasDocImage(self):
        if self.document:
            if self.document["mime_type"] in MIME_TYPES:
                return True
        return False
    
    def _LargestPhoto(self):
        if isinstance(self.photo,dict):
            return self.photo["file_id"]
        if isinstance(self.photo,list):
            max_size = 0
            file_id = ""
            for ele in self.photo:
                cur_size = ele["width"]*ele["height"]
                if cur_size>max_size:
                    file_id = ele["file_id"]
            #
            return file_id
    def __repr__(self):
        return str(self.update_id)
    
    def Send(self,text,markdown=False,reply_markup=None):
        sendMessage = request_format.format(token=TOKEN,metodo="sendMessage")
        sendMessagePars = {"chat_id":self.chat_id,"text":text}
        if markdown:
            sendMessagePars["parse_mode"] = "Markdown"
        if reply_markup:
            sendMessagePars["reply_markup"] = json.dumps(reply_markup)
            
        _=requests.get(sendMessage,sendMessagePars)
        
    def SendPhoto(self,url,caption=None,markdown=False):
        """
        url:direccion en la web de la foto
        caption:descripcion de la foto
        markdown:habilita formato para estilos en caption
        """
        sendImage = request_format.format(token=TOKEN,metodo="sendPhoto")
        sendImagePars = {"chat_id":self.chat_id,"photo":url}
        if caption:
            sendImagePars["caption"] = caption
        if caption and markdown:
            sendImagePars["parse_mode"] = "Markdown"
        _=requests.get(sendImage,sendImagePars)
        
    def SendMediaGroup(self,media):
        "media:array of inputmediaphoto"
        sendArray = request_format.format(token=TOKEN,metodo="sendMediaGroup")
        sendArrayPars = {"chat_id":self.chat_id,"media":json.dumps(media)}
        print(sendArrayPars)
        _=requests.get(sendArray,sendArrayPars)
    
    def answerCallbackQuery(self,text):
        sendArray = request_format.format(token=TOKEN,metodo="answerCallbackQuery")
        sendArrayPars = {"callback_query_id":self.callback_query_id,
                         "text":text}
        _=requests.get(sendArray,sendArrayPars)
        
    def getFilePath(self):
        if self.file_id:
            getFile = request_format.format(token=TOKEN,metodo="getFile")
            getFilePars = {"file_id":self.file_id}
            self.file_path = requests.get(getFile,getFilePars).json()["result"]["file_path"]
            
    def download_file(self):
        filename = path.join(DIR,str(self.update_id))
        download_link = file_format.format(token=TOKEN,file_path=self.file_path)
        local_filename,headers = urlretrieve(download_link,filename)
        return local_filename


##################################################################################


    
def responder(mensaje,funcion):#Para hilos
    "Responde a un mensaje"
    
    #Si contiene una imagen para descargar:
    if mensaje.hasDocImage() or mensaje.hasPhoto():
   
        mensaje.getFilePath()
        local_filename = mensaje.download_file()
        captions,urls = funcion(local_filename)
        remove(local_filename)
        
        #Si se configuraron imagenes de referencia:
        if urls:
            media = PackResponse(captions,urls)
            mensaje.SendMediaGroup(media)
        else:
            mensaje.Send(captions)
    
    #Si el mensaje contiene texto o comandos:
    elif mensaje.mensaje:
        #Enviar tabla de especies
        if mensaje.text=="/especies":
            path = "enviados/tabla_especies.link"
            link = load_text(path)
            mensaje.SendPhoto(link)
            
            
        else:#Enviar boton de instrucciones
            keyboard = [[InlineKeyboardButton("Instrucciones","1")]]
            keyboard = InlineKeyboardMarkup(keyboard)
            text = """Para consultar por una pieza de madera 
                      envíe una imagen"""
            mensaje.Send(text,reply_markup=keyboard)
   
    #Si se presionó un boton previamente enviado
    elif mensaje.callback_query:
        path = "enviados/instrucciones.msg"
        text = load_text(path)   
        mensaje.answerCallbackQuery("Cargando  instrucciones")
        mensaje.Send(text,markdown=True)
    
    
def max_update_id(cola_mensajes,maxId):
    for mensaje in cola_mensajes:
        if mensaje.update_id>maxId:
            maxId = mensaje.update_id
    return maxId

def iniciar_bot(funcion,tiempo_activo=120):
  """
  !Esta funcion solo se debe usar en caso de que la conexion a Telegram se realice atraves de LongPooling
  
  Activa el bot para responder a imagenes con el texto que entrega "funcion"
  funcion: debe recibir el nombre del archivo de una imagen y devolver texto
  
  En el hilo principal,cada TIMEOUT segundos inicia un nuevo pooling para recibir actualizaciones
  Si se recibe una actualizacion se parte en varios mensajes, 
  cada mensaje se procesa(descarga y analisis de imagen,envio de respuesta) en un hilo diferente.
  """

  TIMEOUT=5#S
  updater = LongPoolingUpdate(TIMEOUT)
  hilos = []

  for i in range(tiempo_activo//TIMEOUT):
      cola_mensajes = updater.get()
      updater.last_update_id = max_update_id(cola_mensajes,updater.last_update_id)

      for mensaje in cola_mensajes:
        hilos.append(
                threading.Thread(target=responder,args=(mensaje,funcion)))
        hilos[-1].start()
              
      print(sum([1 for hilo in hilos if hilo.is_alive()])," hilos activos")
      
def deleteWebhook():
  deleteWebhook = request_format.format(token=TOKEN,metodo="deleteWebhook")
  respuesta=requests.get(deleteWebhook)
  print(respuesta)
  
def setWebhook(URL):
  setWebhook = request_format.format(token=TOKEN,metodo="setWebhook")
  Pars = {"url":URL}
  respuesta=requests.get(setWebhook,Pars)
  print(respuesta)

def Webhook_bot(update,funcion):
  updater = WebhookUpdate(update)
  mensaje = updater.get()
  print("id",mensaje.update_id)
  responder(mensaje,funcion)


