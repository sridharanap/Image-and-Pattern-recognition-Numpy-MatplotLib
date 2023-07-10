from pattern import Checker,Circle,Spectrum
checker=Checker(15,5)
print(checker.draw())
checker.show()
circle=Circle(100,50,(50,50))
pattern2=circle.draw()
circle.show()
spectrum=Spectrum(256)
pattern3=spectrum.draw()
spectrum.show()
from generator import ImageGenerator
label_path = './Labels.json'
file_path = './exercise_data/'
gens=ImageGenerator(file_path, label_path, 12, [32, 32, 3], rotation=False, mirroring=False,shuffle=False)
gens.show()

