import random 

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

if __name__ == "__main__":
	#trainingGenerator()
	testGenerator()