set root=C:\Users\krish\Anaconda3

call %root%\Scripts\activate.bat %root%

python xlnet_humor.py
python xlnet_controversial.py
python xlnet_humor_rating.py
python xlnet_offense.py

pause