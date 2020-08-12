#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from TelegramImagenesBot import iniciar_bot
from Inference import LoadedModel


model_filename = "MD_B1_02271216.onnx"
loaded_model = LoadedModel(model_filename)

funcion = loaded_model.predict
iniciar_bot(funcion,tiempo_activo=120)

