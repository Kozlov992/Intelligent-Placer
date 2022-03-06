# Intelligent-Placer

Краткое описание лабораторной "Intelligent-Placer" по курсу "Обработатка и интерпретация сигналов".
____
## Краткое описание

По данной на вход фотографии нескольких расположенных на белом листе бумаги предметов и многоугольнику (сам лист помещен на темную горизонтальную поверхность) понимать, можно ли расположить одновременно все эти предметы на плоскости так, чтобы они влезли в этот многоугольник. Предметы и горизонтальная поверхность, которые могут оказаться на фотографии, заранее известны и находятся в папке objects.
____
## Требования

### **К входным данным**

- Многоугольник должен быть задан замкнутым ломаным контуром, нарисованным темным маркером на белом листе бумаги, сфотографированной вместе с предметами.

- Разрешение подаваемых на вход фотографий: 1620 на 2160

- Высота съемки: от 30 до 50 см

- Объекты должны попадать в кадр целиком

- Угол наклона камеры: до 5 градусов 

- Вращение предметов допускается только вокруг оси, совпадающей с нормалью к горизонтальной поверхности, на которой они находятся
___
### **К выходу**

- Один из двух ответов - "yes"/"no", записанный в желаемый поток вывода (файл или консоль).
___
___

## Алгоритм

### **Начальный вариант**

- Пусть на начальных этапах алгоритм будет решать упрощенный вариант заранее поставленной задачи, а именно - проверять необходимое условие "вмещаемости" предметов в заданный многоугольник (то есть такое, при нарушении которого можно с уверенностью сказать, что интересующие нас предметы в многоугольник не влезут). Формально условие выглядит следующим образом: пусть на плоскости имеется набор плоских объектов <img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\bg_white&space;A=\{a_1,&space;a_2,&space;...,&space;a_n\}\subset&space;2^{\mathbb{R}^2}" title="http://latex.codecogs.com/png.latex?\dpi{110} \bg_white A=\{a_1, a_2, ..., a_n\}\subset 2^{\mathbb{R}^2}" /> c площадями соответственно <img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\bg_white&space;\mu(a_1),&space;\mu(a_2),...,\mu(a_n)" title="http://latex.codecogs.com/png.latex?\dpi{110} \bg_white \mu(a_1), \mu(a_2),...,\mu(a_n)" /> (функционал <img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\bg_white&space;\mu" title="http://latex.codecogs.com/png.latex?\dpi{110} \bg_white \mu" /> ставит в соответствие плоскому предмету его площадь). Известно также, что <img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\bg_white&space;\forall&space;i\neq&space;j\quad&space;\mu(a_i\cap&space;a_j)\equiv0" title="http://latex.codecogs.com/png.latex?\dpi{110} \bg_white \forall i\neq j\quad \mu(a_i\cap a_j)\equiv0" />. Также дан многоугольник <img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\bg_white&space;P" title="http://latex.codecogs.com/png.latex?\dpi{110} \bg_white P" />. Тогда необходимое условие "вмещаемости" объектов <img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\bg_white&space;a_1,...,a_n" title="http://latex.codecogs.com/png.latex?\dpi{110} \bg_white a_1,...,a_n" /> в <img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\bg_white&space;P" title="http://latex.codecogs.com/png.latex?\dpi{110} \bg_white P" /> будет выглядеть следующим образом: <img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\bg_white&space;\displaystyle\sum\limits_{i=1}^n\mu(a_i)\leq\mu(P)" title="http://latex.codecogs.com/png.latex?\dpi{110} \bg_white \displaystyle\sum\limits_{i=1}^n\mu(a_i)\leq\mu(P)" />.
- Если поставленное на площади предметов условие выполнено, то можно перейти к проверке следующего эмпирического условия. Пусть все еще имеется набор <img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\bg_white&space;A=\{a_1,...,a_n\}" title="http://latex.codecogs.com/png.latex?\dpi{110} \bg_white A=\{a_1,...,a_n\}" /> и многоугольник <img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\bg_white&space;P" title="http://latex.codecogs.com/png.latex?\dpi{110} \bg_white P" />. Диаметром <img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\bg_white&space;\text{diam}(M)" title="http://latex.codecogs.com/png.latex?\dpi{110} \bg_white \text{diam}(M)" /> предмета <img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\bg_white&space;M" title="http://latex.codecogs.com/png.latex?\dpi{110} \bg_white M" /> будем называть величину<img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\bg_white&space;\displaystyle\sup_{x,y\in&space;M}\rho(x,&space;y)" title="http://latex.codecogs.com/png.latex?\dpi{110} \bg_white \displaystyle\sup_{x,y\in M}\rho(x, y)" /> (где  <img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\bg_white&space;\rho" title="http://latex.codecogs.com/png.latex?\dpi{110} \bg_white \rho" />  - метрика на плоскости). Тогда новое условие "вмещаемости" предметов примет вид: <img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\bg_white&space;\text{diam}(P)\geq\displaystyle\max_{a\in&space;A}\text{diam}(a)" title="http://latex.codecogs.com/png.latex?\dpi{110} \bg_white \text{diam}(P)\geq\displaystyle\max_{a\in A}\text{diam}(a)" />. Очевидно также, что диаметр многоугольника - это длина одной из линий, соединяющих вершины многоугольника.
- Зададим весьма "усиленное", но простое достаточное условие "вмещаемости" предметов в многоугольник. Найдем прямоугольники <img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\bg_white&space;P_1,&space;...,&space;P_n" title="http://latex.codecogs.com/png.latex?\dpi{110} \bg_white P_1, ..., P_n" />, такие что <img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\bg_white&space;a_i\subseteq&space;P_i" title="http://latex.codecogs.com/png.latex?\dpi{110} \bg_white a_i\subseteq P_i" />, <img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\bg_white&space;P_i=[a^i_0,\,a^i_1]\times&space;[b^i_0,\,&space;b^i_1]" title="http://latex.codecogs.com/png.latex?\dpi{110} \bg_white P_i=[a^i_0,\,a^i_1]\times [b^i_0,\, b^i_1]" /> и прямоугольник <img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\bg_white&space;P^0=[A_1,&space;A_2]\times[B_1,&space;B_2]" title="http://latex.codecogs.com/png.latex?\dpi{110} \bg_white P^0=[A_1, A_2]\times[B_1, B_2]" />, такой что <img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\bg_white&space;P^0\subseteq&space;P" title="http://latex.codecogs.com/png.latex?\dpi{110} \bg_white P^0\subseteq P" />. Если <img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\bg_white&space;\displaystyle\sum\limits_{i=1}^n&space;&space;(a_1^i-a_0^i)\leq&space;A_2-A_1" title="http://latex.codecogs.com/png.latex?\dpi{110} \bg_white \displaystyle\sum\limits_{i=1}^n (a_1^i-a_0^i)\leq A_2-A_1" /> и <img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\bg_white&space;\displaystyle\sum\limits_{i=1}^n&space;(b_1^i-b_0^i)\leq&space;B_2-B_1" title="http://latex.codecogs.com/png.latex?\dpi{110} \bg_white \displaystyle\sum\limits_{i=1}^n (b_1^i-b_0^i)\leq B_2-B_1" /> , то фигуры можно вместить в многоугольник.

### **Возможные улучшения**
- "Усилить" необходимые условия (и/или добавить новые) и "ослабить" достаточные. Текущее достаточное условие в силу своей простоты не будет покрывать огромное количество даже не граничных, а типовых случаев. "Сильные" необходимые условия нужны для того, чтобы отсяеть от исследования на вместимость потенциально простые случаи (учитывая, что асимптотическое время алгоритма отсева меньше, чем у алгоритма исследования на вместимость).
- Сам алгоритм исследования на вместимость предполагается реализовывать в виде жадного алгоритма.
