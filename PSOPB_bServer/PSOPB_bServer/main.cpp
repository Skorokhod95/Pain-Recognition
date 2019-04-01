#include <cstdio>
#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS
//#define PyList_SET_ITEM

#include <iostream>
#include <cmath>
#include <string>
#include <ctime>
#include <vector>
#include <algorithm>
#include <map>
#include <fstream>
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <bitset>
#include "python2.7/Python.h"

using namespace std;
//using namespace std::chrono;

std::vector<double> &operator += (std::vector<double> &a, std::vector<double> &b) {
	vector<double> d(a.size());
	for (int i = 0; i < a.size(); i++) {
		a[i] += b[i];
	}
	return a;
}



double Gaussian(double, double);
double Gaussian(double M, double D)
{
	/*Сложим n случайных чисел, используя стандартный ГСЧ :

	Согласно ЦПТ числа V образуют ряд значений, распределенный по нормальному закону.Эти числа тем лучше описывают нормальный закон,
	чем больше параметр n.На практике n берут равными 6 или 12. Заметим, что закон распределения чисел V имеет математическое
	ожидание mV = n / 2, ?V = sqrt(n / 12).Поэтому он является смещенным относительно заданного произвольного.
	С помощью формулы z = (V – mV) / ?V нормализуем этот ряд.Получим нормализованный закон нормального распределения чисел Z.То есть mz = 0, ?z = 1.
	Формулой(сдвиг на mx и масштабирование на ?x) преобразуем ряд Z в ряд x : x = z · ?x + mx.*/

	double V = 0, z, mV, sigmaV;
	int i, n = 12;

	mV = n / 2.;
	sigmaV = sqrt(n / 12.);

	for (i = 0; i < n; i++)
		V += (rand() / double(RAND_MAX));

	z = (V - mV) / sigmaV;
	return z * D + M;
}

double Cauchy();
double Cauchy()
{
	return tan(M_PI*((rand() / double(RAND_MAX)) - 0.5));
}

class Py {
public:
	//to mass call
	/*PyObject * myModuleString;
	PyObject* myModule;
	PyObject* myFunction;
	PyObject *mylist_Dim, *mylist_Pop;
	PyObject* args;
	PyObject* myResult;
	PyObject* val;*/

	double result;
	int size, Dimention;
	int counter = 0;
	string start = "module", end = ".py", newone_write, newone_read;
	~Py() {};

	void create_py(int);
	vector<double> fitness(vector<vector<bool>>, int, int);
};

void Py::create_py(int counter) {
	ifstream fin;
	fin.open("/home/askorokhod/projects/PSOPB_bServer/module121.py");
	string content;
	copy(istreambuf_iterator<char>(fin), istreambuf_iterator<char>(), back_inserter(content));

	newone_write = start + to_string(counter) + end;
	newone_read = start + to_string(counter);

	ofstream fout;
	fout.open(newone_write);
	fout << content;
	//cout << content;

	fout.close();
	fin.close();
}

vector<double> Py::fitness(vector<vector<bool>> array, int si, int pop_size)
{
	vector<double> results;
	results.reserve(pop_size);

	PyObject * myModuleString;
	PyObject* myModule;
	PyObject* myFunction;
	PyObject *mylist_Dim, *mylist_Pop;
	PyObject* args;
	PyObject* myResult;
	PyObject* val;


	//setenv("PYTHONPATH", ".", 1);

	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('/home/askorokhod/projects/PSOPB_bServer/bin/x64/Debug/')");
	//cout << 401 << '\n';
	myModuleString = PyString_FromString((newone_read).c_str());
	if (!myModule) {
		PyErr_Print();
	}
	//cout << 45 << '\n';
	myModule = PyImport_Import(myModuleString);
	//cout << "Module" << myModule << '\n';
	myFunction = PyObject_GetAttrString(myModule, (char*)"func");
	//cout << "function" << myFunction << '\n';

	mylist_Pop = PyList_New(pop_size);

	//cout << 51 << '\n';

	for (int j = 0; j < pop_size; j++) {
		mylist_Dim = PyList_New(si);//2- of list
		for (size_t i = 0; i != si; ++i) {
			PyList_SET_ITEM(mylist_Dim, i, PyBool_FromLong(array[j][i]));
		}
		PyList_SET_ITEM(mylist_Pop, j, mylist_Dim);
	}
	args = PyTuple_Pack(1, mylist_Pop);

	myResult = PyObject_CallObject(myFunction, args);

	for (int y = 0; y < pop_size; y++) {
		results[y] = PyFloat_AsDouble(PyList_GetItem(myResult, y));
	}


	Py_DECREF(myModuleString);
	Py_DECREF(myModule);
	Py_DECREF(myFunction);
	Py_DECREF(mylist_Dim);
	Py_DECREF(mylist_Pop);
	Py_DECREF(args);
	Py_DECREF(myResult);



	return results;
}


class PSOPB_b
{
public:
	//Конструктор
	//По умолчанию размерность задачи равна 1, радиус поискового пространства равен 1, размер популяции равен 30, количество поколений 10
	PSOPB_b() {
		Dimention = 77;
		RightBorder = 10;
		LeftBorder = -10;
		NumberOfGenerations = 50;
		PopulationSize = 100;

		//константы
		c[1] = 2.05;	c[2] = 2.05;
		c[11] = 1.367;	c[12] = c[11]; c[13] = c[11];
		//------------------------------------
		pc["max"] = 0.02; pc["min"] = 0.005;
		//------------------------------------
		epsilon["max"] = 0.5; epsilon["min"] = 0.3;
		//------------------------------------
		sigma["max"] = 1; sigma["min"] = 0.1;
		//------------------------------------
	}
	~PSOPB_b() {};


	//Переменные
	double RightBorder, LeftBorder;
	Py py;

	//Функции
	/*void PrintSomething(ofstream &fout, int t) {
		fout << "Поколение " << t << '\n';
		fout << "Лучший паразит	" << "Лучший хозяин	" << "Глобальный лучший" << '\n';
		fout << ValuePgP << '\t' << ValuePgH << '\t' << ValueGB << '\n';
		fout << "Рой паразитов: " << '\n';
		for (int i = 0; i < Dimention; i++) {
			fout << "X" << i << '\t';
		}
		fout << "Значение в точке" << '\n';
		for (int i = 0; i < FitnessParasiteSwarm.size(); i++) {
			for (auto &j : ParasiteSwarm[i]) {
				fout << j << '\t';
			}
			fout << FitnessParasiteSwarm[i] << '\n';
		}
		fout << "Рой хозяев" << '\n';
		for (int i = 0; i < FitnssHostSwarm.size(); i++) {
			for (auto &j : HostSwarm[i]) {
				fout << j << '\t';
			}
			fout << FitnssHostSwarm[i] << '\n';
		}
	}*/
	void Initialization(const int&, const double&, const int&, const int&);
	void psopb_b();
	vector<bool> GlobalBest;
	double ValueGB;

	int TypeOfFunction;

private:
	//Переменные
	int PopulationSize, NumberOfGenerations, Dimention, nn;
	vector<vector<double>> Velocity, ParasiteSwarm_v, HostSwarm_v;
	vector<vector<bool>> ParasiteSwarm, HostSwarm;
	vector<double>  FitnessParasiteSwarm, FitnssHostSwarm, VMax, Pc, Pn;
	map <int, double> c;
	map <string, double> pc, epsilon, sigma;
	map <string, vector<double>> ValuePi;
	map<string, vector<bool>> pg;
	map <string, vector<vector<bool>>> pi;
	const double lyambda = 0.729, gamma = 0.5, w = 0.7968;
	double ValuePgH, ValuePgP;
	vector<double> StorageValuePgP;
	//
	vector<vector <bool>> temp_S;
	vector<double> temp_F;
	map<double, vector<bool>> Host, Parasit;
	vector<vector<bool>> MemoryWorst, MemoryBest;
	vector<double> MemoryWorstFitness, MemoryBestFitness;
	map <string, vector<vector<double>>> New;
	vector<bool> Pc_b, Pn_b;
	int for_new_one = 0;
	//Функции
	void RandonGeneration();
	void ValueEntry_mass(vector<vector<bool>>&, vector<double>&, int);
	void ValueEntry_single(vector<bool>&, double&);
	void UpdateValues();
	void Bests();
	void Sort();
	void ReplaceWorst(int&);
	void VelocityMutation();
	void PositionMutation();
	void GBest();
	void to_bool(vector<double>&, vector<bool>&);
};

//Функция перевода в булевскую строку
void PSOPB_b::to_bool(vector<double>& x, vector<bool>&x_b)
{
	for (int d = 0; d<Dimention; d++) {
		x[d] = 1 / (1 + exp(-x[d]));
		if (((double)(rand()) / RAND_MAX)>x[d]) {
			x_b[d] = 1;
		}
		else {
			x_b[d] = 0;
		}
	}
}

void PSOPB_b::psopb_b() {

	//Случайная инициализация
	RandonGeneration();
	ValueEntry_mass(ParasiteSwarm, FitnessParasiteSwarm, PopulationSize);
	ValueEntry_mass(HostSwarm, FitnssHostSwarm, PopulationSize);
	ValuePgP = FitnessParasiteSwarm[0];
	ValuePgH = FitnssHostSwarm[0];
	ValueGB = min(ValuePgH, ValuePgP);
	Sort();
	ValuePi["Parasite"].resize(PopulationSize);
	for (int i = 0; i < PopulationSize; i++) {
		ValuePi["Parasite"][i] = FitnessParasiteSwarm[i];
	}
	ValuePi["Host"] = FitnssHostSwarm;
	pi["Parasite"] = ParasiteSwarm;
	pi["Host"] = HostSwarm;
	pg["Parasite"] = ParasiteSwarm[0];
	pg["Host"] = HostSwarm[0];
	for (int t = 0; t < NumberOfGenerations; t++) {
		Bests();
		//cout << "Bests	" << t << '\n';
		StorageValuePgP[t] = ValuePgP;
		GBest();
		//cout << "GBests	" << t << '\n';
		UpdateValues();
		//cout << "UpdateValue	" << t << '\n';
		Sort();
		//cout << "Sort	" << t << '\n';
		for (int g = 0; g < Dimention; g++) {
			cout << GlobalBest[g] << '\t';
		}
		cout << '\n';
		cout << ValueGB << '\n';
		ReplaceWorst(t);
		//cout << "ReplaceWorst	" << t << '\n';
		if (t > nn && ValuePgP == StorageValuePgP[t - nn]) {
			VelocityMutation();
			//cout << "VelocityMutation	" << t << '\n';
		}
		PositionMutation();
		//cout << "PositionMutation	" << t << '\n';
	}
	GBest();
}

//Поиск глобального лучшего
void PSOPB_b::GBest() {

	//#pragma omp parallel for
	for (int i = 0; i < PopulationSize; i++) {
		if (FitnessParasiteSwarm[i]<=FitnssHostSwarm[i]) {
			if (ValueGB <= FitnssHostSwarm[i]) {
				ValueGB = FitnssHostSwarm[i];
				GlobalBest = HostSwarm[i]; 
			}
		}
		else {
			if (ValueGB < FitnessParasiteSwarm[i]) {
				ValueGB = FitnessParasiteSwarm[i];
				GlobalBest = ParasiteSwarm[i];
			}
		}
	}

}

//Реализация оператора мутации
void PSOPB_b::VelocityMutation() {
	int Counter = 0, Place;
	
	double MutationProbability = 0.005;

	//Мутация скорости
//#pragma omp parallel for
	for (int i = 0; i < PopulationSize * 2; i++) {
		if (((double)(rand()) / RAND_MAX) < MutationProbability) {
			for (int j = 0; j < Dimention; j++) {
				if (((double)(rand()) / RAND_MAX) < 0.5) {
					Velocity[i][j] = 0.5 * VMax[j] * ((double)(rand()) / RAND_MAX);
				}
				else {
					Velocity[i][j] = -0.5 * VMax[j] * ((double)(rand()) / RAND_MAX);
				}
			}
			if (i >= PopulationSize) {
				to_bool(Velocity[i], HostSwarm[i - PopulationSize]);
				ValueEntry_single(HostSwarm[i - PopulationSize], FitnssHostSwarm[i - PopulationSize]);
			}
			else {
				to_bool(Velocity[i], ParasiteSwarm[i]);
				ValueEntry_single(ParasiteSwarm[i], FitnessParasiteSwarm[i]);
			}
			Counter++;
		}
	}

	if (Counter == 0) {
		Place = rand() % PopulationSize * 2;
//#pragma omp parallel for
		for (int j = 0; j < Dimention; j++) {
			if (((double)(rand()) / RAND_MAX) < 0.5) {
				Velocity[Place][j] = 0.5 * VMax[j] * ((double)(rand()) / RAND_MAX);
			}
			else {
				Velocity[Place][j] = -0.5 * VMax[j] * ((double)(rand()) / RAND_MAX);
			}
		}
		if (Place >= PopulationSize) {
			to_bool(Velocity[Place], HostSwarm[Place - PopulationSize]);
			ValueEntry_single(HostSwarm[Place - PopulationSize], FitnssHostSwarm[Place - PopulationSize]);
		}
		else {
			to_bool(Velocity[Place], ParasiteSwarm[Place]);
			ValueEntry_single(ParasiteSwarm[Place], FitnessParasiteSwarm[Place]);
		}
	}
}

//Позиционная мутация
void PSOPB_b::PositionMutation() {
	int Place = rand() % Dimention;
	

#pragma omp parallel for
	for (int i = 0; i < Dimention; i++) {
		if (i == Place) {
			Pn[i] = pg["Parasite"][i] + (RightBorder - LeftBorder)*Gaussian(0, 1);
			Pc[i] = pg["Parasite"][i] + Cauchy();
			if (Pn[i] < LeftBorder) {
				Pn[i] = min(RightBorder, 2 * LeftBorder - Pn[i]);
			}
			if (Pn[i]>RightBorder) {
				Pn[i] = max(LeftBorder, 2 * RightBorder - Pn[i]);
			}
			if (Pc[i] < LeftBorder) {
				Pc[i] = min(RightBorder, 2 * LeftBorder - Pc[i]);
			}
			if (Pc[i]>RightBorder) {
				Pc[i] = max(LeftBorder, 2 * RightBorder - Pc[i]);
			}
		}
		else {
			Pn[i] = pg["Parasite"][i];
			Pc[i] = pg["Parasite"][i];
		}
	}
	to_bool(Pc, Pc_b);
	to_bool(Pn, Pn_b);
	double intermediatePc;
	ValueEntry_single(Pc_b, intermediatePc);
	double intermediatePn;
	ValueEntry_single(Pn_b, intermediatePn);

	if (intermediatePc>intermediatePn) {
		if (ValuePgP > intermediatePn) {
			ValuePgP = intermediatePn;
			pg["Parasite"] = Pn_b;
		}
	}
	else {
		if (ValuePgP > intermediatePc) {
			ValuePgP = intermediatePc;
			pg["Parasite"] = Pc_b;
		}
	}
}

//Удаление худших индивидов 
void PSOPB_b::ReplaceWorst(int& t) {
	int Epsilon = (epsilon["max"] - (epsilon["max"] - epsilon["min"])*((double)t / NumberOfGenerations)) * PopulationSize;
	int Gamma = PopulationSize * 0.05;
	
	MemoryWorst.resize(Epsilon);
	MemoryBest.resize(Epsilon);
	MemoryWorstFitness.resize(Epsilon);
	MemoryBestFitness.resize(Epsilon);
	//Обмен худшими и лучшими индивидами в роях
	//Sort();

//#pragma omp parallel for
	for (int i = 0; i < Epsilon; i++) {
		MemoryBest[i] = HostSwarm[PopulationSize - i - 1];
		MemoryBestFitness[i] = FitnssHostSwarm[PopulationSize - i - 1];
		MemoryWorst[i] = ParasiteSwarm[PopulationSize - i - 1];
		MemoryWorstFitness[i] = FitnessParasiteSwarm[PopulationSize - i - 1];
	}
//#pragma omp parallel for
	for (int i = 0; i < Epsilon; i++) {
		HostSwarm[i] = MemoryWorst[i];
		FitnssHostSwarm[i] = MemoryWorstFitness[i];
		ParasiteSwarm[PopulationSize - 1 - i] = MemoryBest[i];
		FitnessParasiteSwarm[PopulationSize - 1 - i] = MemoryBestFitness[i];
	}

	//Удаление худших индивидов из роя хозяев
//#pragma omp parallel for
	for (int j = 0; j < Gamma; j++) {
		for (int d = 0; d < Dimention; d++) {
			if (((double)(rand()) / RAND_MAX) <= 0.5) {
				HostSwarm[j][d] = LeftBorder + (RightBorder - LeftBorder)*((double)(rand()) / RAND_MAX);
			}
			else {
				HostSwarm[j][d] = GlobalBest[d] + (RightBorder - LeftBorder) *
					(0.3 - (0.25 * (double)t) / NumberOfGenerations) * (2 * ((double)(rand()) / RAND_MAX) - 1);
			}
		}
	}
	if (Gamma > 0) {
		ValueEntry_mass(HostSwarm, FitnssHostSwarm, Gamma);
	}
}

//Сортировка роя паразитов и хозяев по значению пригодности индивида
void PSOPB_b::Sort() {
	int j = 0, d = PopulationSize - 1;

	//#pragma omp parallel for
	for (int i = 0; i < PopulationSize; i++) {
		Host[FitnssHostSwarm[i]] = HostSwarm[i];
		Parasit[FitnessParasiteSwarm[i]] = ParasiteSwarm[i];
	}

	for (auto item : Parasit) {
		ParasiteSwarm[j] = item.second;
		FitnessParasiteSwarm[j] = item.first;
		j++;
	}
	for (auto jtem : Host) {
		HostSwarm[d] = jtem.second;
		FitnssHostSwarm[d] = jtem.first;
		d--;
	}

	Host.clear();
	Parasit.clear();
}

//Поиск лучших значений в популяции
void PSOPB_b::Bests() {
	Sort();
	//Поиск лучших для каждого индивида
#pragma omp parallel for
	for (int i = 0; i < PopulationSize; i++) {
		if (FitnessParasiteSwarm[i]<ValuePi["Parasite"][i]) {
			ValuePi["Parasite"][i] = FitnessParasiteSwarm[i];
			pi["Parasite"][i] = ParasiteSwarm[i];
		}
	}


	//Поиск лучшего для каждого роя
	if (ValuePgH > FitnssHostSwarm[PopulationSize - 1]) {
		ValuePgH = FitnssHostSwarm[PopulationSize - 1];
		pg["Host"] = HostSwarm[PopulationSize - 1];
	}
	if (ValuePgP > FitnessParasiteSwarm[0]) {
		ValuePgP = FitnessParasiteSwarm[0];
		pg["Parasite"] = ParasiteSwarm[0];
	}
}

//Обновление значений скорости и местоположения
void PSOPB_b::UpdateValues() {
#pragma omp parallel for
	for (int i = 0; i < PopulationSize; i++) {
		for (int j = 0; j < Dimention; j++) {
			//Обновление скорости роя паразитов
			Velocity[i][j] = lyambda * (w*Velocity[i][j] + c[1] * ((double)(rand()) / RAND_MAX)*(pi["Parasite"][i][j] - ParasiteSwarm[i][j]) +
				c[2] * ((double)(rand()) / RAND_MAX)*(pg["Parasite"][j] - ParasiteSwarm[i][j]));

			//Обновление скорости роя хозяев
			if (ValuePgH >= ValuePgP) {
				Velocity[i + PopulationSize][j] = lyambda * (Velocity[i + PopulationSize][j] + c[11] * ((double)(rand()) / RAND_MAX)*(pi["Host"][i][j] - HostSwarm[i][j]) +
					c[12] * ((double)(rand()) / RAND_MAX)*(pg["Host"][j] - HostSwarm[i][j]) +
					c[13] * ((double)(rand()) / RAND_MAX)*(pg["Parasite"][j] - HostSwarm[i][j]));
			}
			else {
				Velocity[i + PopulationSize][j] = lyambda * (Velocity[i + PopulationSize][j] + c[1] * ((double)(rand()) / RAND_MAX)*(pi["Host"][i][j] - HostSwarm[i][j]) +
					c[2] * ((double)(rand()) / RAND_MAX)*(pg["Host"][j] - HostSwarm[i][j]));
			}
		}
		//Обновление положения роя хозяев
		to_bool(Velocity[i + PopulationSize], HostSwarm[i]);
		//Обновление положения роя паразитов
		to_bool(Velocity[i], ParasiteSwarm[i]);
	}
	//Обновление значений пригодности каждого индивида
	ValueEntry_mass(ParasiteSwarm, FitnessParasiteSwarm, PopulationSize);
	ValueEntry_mass(HostSwarm, FitnssHostSwarm, PopulationSize);
}

//Рассчет значений функции пригодности для индивидов
void PSOPB_b::ValueEntry_mass(vector<vector<bool>>& Swarm, vector<double>& FitnessValue, int Number) {
	py.create_py(for_new_one);
	//#pragma omp parallel for
	if (Number < Swarm.size()) {
		temp_S = Swarm;
		temp_S.resize(Number);
		temp_F = py.fitness(temp_S, Dimention, Number);
		for (int i = 0; i < Number; i++) {
			FitnessValue[i] = temp_F[i];
		}
	}
	else {
		FitnessValue = py.fitness(Swarm, Dimention, Number);
	}
	for_new_one++;
}

void PSOPB_b::ValueEntry_single(vector<bool>& Swarm, double& FitnessValue) {
	py.create_py(for_new_one);
	temp_S.resize(1);
	temp_S[0] = Swarm;
	temp_F = py.fitness(temp_S, Dimention, 1);
	FitnessValue = temp_F[0];
	for_new_one++;
}

//Задание входных параметров пользователем
void PSOPB_b::Initialization(const int& dimention, const double& radius, const int& number_of_generation, const int& population_size) {
	Dimention = dimention;
	NumberOfGenerations = number_of_generation;
	PopulationSize = population_size;
	RightBorder = radius;
	LeftBorder = -radius;
}

//Заполнение векторов роя паразитов и роя хозяев случайными числами 
void PSOPB_b::RandonGeneration() {
	Velocity.clear();
	Velocity.resize(PopulationSize * 2);
	//Случайная инициализация 
	ParasiteSwarm.clear();
	HostSwarm.clear();
	ParasiteSwarm.resize(PopulationSize);
	HostSwarm.resize(PopulationSize);
	ParasiteSwarm_v.clear();
	HostSwarm_v.clear();
	ParasiteSwarm_v.resize(PopulationSize);
	HostSwarm_v.resize(PopulationSize);
	Pc_b.reserve(Dimention);
	Pn_b.reserve(Dimention);
	for (int i = 0; i < PopulationSize; i++) {
		for (int j = 0; j < Dimention; j++) {
			ParasiteSwarm_v[i].push_back((double)rand() / RAND_MAX * (RightBorder - LeftBorder) + LeftBorder);
			HostSwarm_v[i].push_back((double)rand() / RAND_MAX * (RightBorder - LeftBorder) + LeftBorder);
			Velocity[i].push_back(0);
			Velocity[i + PopulationSize].push_back(0);
			ParasiteSwarm[i].push_back(0);
			HostSwarm[i].push_back(0);
		}

		to_bool(ParasiteSwarm_v[i], ParasiteSwarm[i]);
		to_bool(HostSwarm_v[i], HostSwarm[i]);
	}

	FitnessParasiteSwarm.clear();
	FitnssHostSwarm.clear();
	FitnessParasiteSwarm.resize(PopulationSize);
	FitnssHostSwarm.resize(PopulationSize);

	VMax.clear();
	VMax.resize(Dimention);
	for (int i = 0; i < Dimention; i++) {
		VMax.push_back(0.25*(RightBorder - LeftBorder));
	}

	nn = 0.05*PopulationSize;
	StorageValuePgP.resize(NumberOfGenerations);
	Pn.resize(Dimention);
	Pc.resize(Dimention);
}

int main()
{
	setlocale(LC_CTYPE, "");
	srand(time(0));
	setenv("PYTHONPATH", ".", 1);
	Py_Initialize();

	ofstream fout, Statistics;
	fout.open("/home/askorokhod/projects/PSOPB_bServer/out.txt", ios_base::app);


	int Dimention, NumberOfGenerations, PopulationSize;
	double Radius=1;
	bool label;
	double mean;
	map<string, map<double, vector<double>>> Extreme;

	cout << "Хотетие оставить настройки алгоритма по умолчанию? 1/0: ";
	cin >> label;

	PSOPB_b psopb_b;
	if (label != 1) {
		cout << "Введите размерность задачи: ";
		cin >> Dimention;
		cout << "Введите количество поколений (точка останова): ";
		cin >> NumberOfGenerations;
		cout << "Введите количество индивидов (размер популяции): ";


		psopb_b.Initialization(Dimention,Radius, NumberOfGenerations, PopulationSize);
	}


	vector<double> ValuesOfGB(1);
	double dispersion = 0, MaxGB, MinGB;


	int start = clock();
	for (int g = 0; g < 1; g++) {
		psopb_b.psopb_b();

		ValuesOfGB[g] = psopb_b.ValueGB;
		fout << "TEST №" << g << '\t' << "Best solution = " << ValuesOfGB[g] << '\n';
		for (int m = 0; m < 77; m++) {
			fout << psopb_b.GlobalBest[m] << '\t';
		}
		fout << '\n';
	}
	fout << '\n' << "time" << '\t' << (clock() - start) / 1000.0 << '\n';


	fout.close();
	Py_Finalize();
	return 0;
}