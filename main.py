import customtkinter as ctk
from gui import RAGApp

if __name__ == "__main__":
    root = ctk.CTk()  # Use customtkinter's CTk instead of tkinter's Tk
    app = RAGApp(root)
    root.mainloop()
