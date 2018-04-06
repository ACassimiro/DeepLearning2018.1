import random 
import math

def trainingGenerator():
	f1=open("cubeSample1.txt", "r")
	f = f1.readlines()
	i = 0
	for line in f:
		x = line.split()

		f2 = open("Noise " + str(i+1)+".txt", "w")

		for j in range(50):
			(a,b,c) = (float(x[0]), float(x[1]), float(x[2]))
			a += random.uniform(-0.1, 0.1)
			b += random.uniform(-0.1, 0.1)
			c += random.uniform(-0.1, 0.1)
			l = [str(a), str(b), str(c)] + x[3:]
			s = ' '.join(l)
			s += '\n'

			f2.write(s)

		f2.close()
		i+=1

	f1.close()

def testGenerator():
	f1=open("cubeSample1.txt", "r")
	f = f1.readlines()
	f2 = open("TestFile.txt", "w")


	for i in range(20):
		line = random.choice(f)
		x = line.split()

		
		(a,b,c) = (float(x[0]), float(x[1]), float(x[2]))
		a += random.uniform(-0.1, 0.1)
		b += random.uniform(-0.1, 0.1)
		c += random.uniform(-0.1, 0.1)
		l = [str(a), str(b), str(c)] + x[3:]
		s = ' '.join(l)
		s += '\n'

		f2.write(s)

	f2.close()
	f1.close()

def traningXORGenerator():
	f1=open("XORSample.txt", "r")
	f = f1.readlines()
	i = 0
	f2 = open("NoiseXOR.txt", "w")

	for line in f:
		x = line.split()
		for j in range(10):
			(a,b) = (float(x[0]), float(x[1]))
			a += random.uniform(-0.1, 0.1)
			b += random.uniform(-0.1, 0.1)
			l = [str(a), str(b), x[2]]
			s = ' '.join(l)
			s += '\n'

			f2.write(s)

		
	f2.close()
	f1.close()

def testXORGenerator():
	f1=open("XORSample.txt", "r")
	f = f1.readlines()
	f2 = open("TestXORFile.txt", "w")


	for i in range(10):
		line = random.choice(f)
		x = line.split()

		
		(a,b) = (float(x[0]), float(x[1]))
		a += random.uniform(-0.1, 0.1)
		b += random.uniform(-0.1, 0.1)
		l = [str(a), str(b), x[2]]
		s = ' '.join(l)
		s += '\n'

		f2.write(s)

	f2.close()
	f1.close()

def trainingSinGenerator():
	f1=open("trainingSin.txt", "w")
	#f1=open("testSin.txt", "w")
	
	for i in range(5000):
		x = random.uniform(0.1, 4)
		s =''
		s += str(x) + ' ' + str(math.sin(x*math.pi)/(x*math.pi)) + '\n'
		f1.write(s)
	f1.close()


if __name__ == "__main__":
	#trainingGenerator()
	#testGenerator()
	#traningXORGenerator()
	#testXORGenerator()
	trainingSinGenerator()