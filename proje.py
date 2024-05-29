
# Modelinizi yüklemek için gerekli kütüphaneler
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer, AdamW, get_scheduler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.nn import Module
import numpy as np
from datasets import Dataset, DatasetDict, load_metric
from torch.utils.data import DataLoader
import torch.cuda
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets  # eklenen satır
from IPython.display import display  # eklenen satır
import tkinter as tk
from tkinter import ttk
from tkinter import font 
from tkinter.constants import DISABLED
import smtplib
from email.mime.text import MIMEText

ref = 'dbmdz/bert-base-turkish-128k-uncased'
tokenizer = AutoTokenizer.from_pretrained(ref)
etiket_map = {"Irkçılık": 0, "Cinsiyetçilik": 1, "Hakaret": 2, "Küfür": 3, "Normal": 4}
etiketler = {
    0: "Irkçılık",
    1: "Cinsiyetçilik",
    2: "Hakaret",
    3: "Küfür",
    4: "Normal"
}
class BertBiltsm(nn.Module):
    def __init__(self, ref, etiket_say):
        super(BertBiltsm, self).__init__()
        self.num_labels = etiket_say
        self.bert = AutoModel.from_pretrained(ref)
        self.bilstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(256, etiket_say)
        self.softmax = nn.Softmax(dim=1)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        lstm_output, _ = self.bilstm(pooled_output.unsqueeze(1).repeat(1, input_ids.size(1), 1))
        lstm_output = self.dropout(lstm_output)
        logits = self.fc(lstm_output[:, -1, :])
        probabilities = self.softmax(logits)
        loss = None
        if labels is not None:
            loss = self.loss_func(logits, labels)
            return probabilities, loss
        else:
            return probabilities

class SiberZorbalikModel(BertBiltsm):
    def __init__(self, ref, etiket_say):
        super(SiberZorbalikModel, self).__init__(ref, etiket_say)

model = SiberZorbalikModel(ref=ref, etiket_say=5)
model.load_state_dict(torch.load('model'))
model.eval()
# Model ve tokenizer yükleme

tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-turkish-128k-uncased')
def center_window(window, width=400, height=300):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    window.geometry(f'{width}x{height}+{x}+{y}')
def classify(text):
    # Tokenize edilen metni ve dikkat maskesini al
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=120, padding=True)
    # 'token_type_ids' girdisini kaldır
    inputs.pop("token_type_ids", None)
    
    with torch.no_grad():
        # Modeli çağırırken sadece gerekli argümanları kullan
        predictions = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    class_id = torch.argmax(predictions, dim=1).item()
    return class_id

def on_analyze():
    text = text_entry.get("1.0", "end-1c").strip()  # Metni al ve boşlukları kaldır
    if text == "" or text == default_text_main:  # Eğer metin kutusu boşsa veya varsayılan metin hala duruyorsa
        result_label.config(text="Lütfen analiz edilecek bir metin girin.")
        return

    class_id = classify(text)
    etiket = etiketler.get(class_id, "Bilinmeyen")

    if etiket == "Normal":
        result_label.config(text="Mesaj gönderildi.")
    else:
        result_label.config(text=f"{etiket} içerdiği için mesaj gönderilemedi.")
root = tk.Tk()
root.title("Zorbalık Tespit Yazılımı")
center_window(root, 400, 300)
root.configure(bg='#D2691E')  # Arka plan rengi

# Fontlar
title_font = font.Font(family="Helvetica", size=16, weight="bold")
text_font = font.Font(family="Helvetica", size=12)
button_font = font.Font(family="Helvetica", size=12, weight="bold")

# Başlık
title_label = tk.Label(root, text="Analiz Yapılacak Metin", font=title_font, bg='#D2691E')
title_label.pack(pady=10)
def add_right_click_menu(widget):
    def on_right_click(event):
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Kes", command=lambda: widget.event_generate('<<Cut>>'))
        menu.add_command(label="Kopyala", command=lambda: widget.event_generate('<<Copy>>'))
        menu.add_command(label="Yapıştır", command=lambda: widget.event_generate('<<Paste>>'))
        menu.tk_popup(event.x_root, event.y_root)
    widget.bind("<Button-3>", on_right_click)  # Windows için
    widget.bind("<Button-2>", on_right_click)  # Mac için

def on_entry_click(event, entry, default_text):
    """Metin kutusuna tıklandığında varsayılan metni siler."""
    if entry.get("1.0", "end-1c") == default_text:
        entry.delete("1.0", "end")
# Metin giriş kutusu
default_text_main = "Zorbalık tespiti için metin giriniz..."
text_entry = tk.Text(root, height=5, width=40, font=text_font, bg='#FFEBCD')
text_entry.insert("1.0", default_text_main)
text_entry.bind("<FocusIn>", lambda event: on_entry_click(event, text_entry, default_text_main))
text_entry.pack(pady=10)
add_right_click_menu(text_entry)
def open_help_window():
    help_window = tk.Toplevel(root)
    help_window.configure(bg='#FFD700')
    help_window.title("Yardım ve Kullanım Kılavuzu")
    help_text = tk.Text(help_window, height=10, width=50, wrap="word", state=DISABLED)
    help_text.pack(pady=10, padx=10)
    center_window(help_window, 400, 200)
    # Metni ekle
    help_content = """
    Bu zorbalık tespit yazılımıdır.
    Girdiğiniz cümleyi analiz et butonuna bastığınızda mesaj gönderilmeye uygunsa mesaj gönderildi yazıcak eğer uygun değilse hangi zorbalık çeşidini kullandıysanız o yüzden mesajınız gönderilmeyecek.
    Kullanım klavuzu:
    Geri bildirim:Hatalı tespitler ya da başka gördüğünüz hataların bildirildiği yer.
    Yardım:Aklınıza takılan sorular için bakabileceğiniz yer.
    Analiz et:Girdiğiniz metini model ile test edip bir çıktı almayı sağlar.
    Daha fazla yardıma ihtiyacınız varsa, sorularınızı yakuperenermurat@gmail.com adresine gönderebilirsiniz.
    """
    help_text.configure(state='normal',  bg='#FFEBCD')
    help_text.insert(tk.END, help_content)
    help_text.configure(state='disabled')
def open_feedback_window():
    default_text_feedback = "Hatalı tespitleri buraya girerek bize yardımcı olabilirsiniz"
    feedback_window = tk.Toplevel(root)
    feedback_window.configure(bg='#228B22')
    feedback_window.title("Geri Bildirim/Öneri")
    center_window(feedback_window, 400, 200)
    feedback_entry = tk.Text(feedback_window, height=5, width=40, bg='#FFEBCD')
    feedback_entry.insert("1.0", default_text_feedback)
    feedback_entry.bind("<FocusIn>", lambda event: on_entry_click(event, feedback_entry, default_text_feedback))
    feedback_entry.pack(pady=10)
    add_right_click_menu(feedback_entry)
    # Geri bildirim gönderildiğinde gösterilecek etiket
    feedback_label = tk.Label(feedback_window, text="", bg='#228B22')
    feedback_label.pack(pady=5)

    def send_feedback():
        feedback = feedback_entry.get("1.0", "end-1c").strip()
        if feedback == default_text_feedback or not feedback:
            feedback_label.config(text="Lütfen geçerli bir geri bildirim girin.")
            return

        # E-posta mesajını oluştur
        msg = MIMEText(feedback)
        msg["Subject"] = "Yeni Geri Bildirim"
        msg["From"] = "mahmutkaraeski259@gmail.com"
        msg["To"] = "yakuperenermurat@gmail.com"

        try:
            # Gmail SMTP sunucusuna bağlan
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login("mahmutkaraeski259@gmail.com", "mgpd gcfu msho owbf")
                server.send_message(msg)

            feedback_label.config(text="Geri bildirim gönderildi. Teşekkürler!")
        except Exception as e:
            feedback_label.config(text=f"Gönderme hatası: {e}")
        # Gönder butonu
    send_button = tk.Button(feedback_window, text="Gönder", command=send_feedback, bg='#D2691E', font=button_font, borderwidth=0, relief='solid')
    send_button.pack(pady=5)
# Yardım Butonu
help_button = tk.Button(root, text="Yardım", command=open_help_window, bg='#D2691E', font=button_font, borderwidth=0, relief='solid')
help_button.place(x=300, y=270)

# Geri Bildirim Butonu
feedback_button = tk.Button(root, text="Geri Bildirim", command=open_feedback_window, bg='#D2691E', font=button_font, borderwidth=0, relief='solid')
feedback_button.place(x=0, y=270)

# Analiz butonu
analyze_button = tk.Button(root, text="Analiz Et", command=on_analyze, bg='#D2691E', font=button_font, borderwidth=0, relief='solid')
analyze_button.pack(pady=10)

# Stil özelleştirmeleri
style = ttk.Style()
style.configure('TButton', font=button_font, padding=6, background='#D2691E', borderwidth=0)
style.configure('TLabel', background='#D2691E')

# Sonuç etiketi
result_label = tk.Label(root, text="", font=text_font, bg='#D2691E')
result_label.pack(pady=10)

# GUI'yi çalıştır
root.mainloop()