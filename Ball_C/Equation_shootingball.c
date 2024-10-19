float shooterV1(float h, float distance, float angle, float alpha)
{
	theta = PI*angle/180;
//	V = sqrt((9.8*pow(distance,2))/(2*h));
//	V = distance/(cos(theta)*(sqrt(2*h)/g));
	V = (distance*0.47)/(m*cos(theta));
	return V;
}
float shooterV2(float h, float distance, float angle ) //good
{
	theta = PI*angle/180;
	V = sqrt((4.9*pow(distance,2)*pow(cos(theta),2))/(h + distance*sin(theta)*cos(theta)));
	return V;
}
float shooterV3(float h, float distance, float angle, float h_end)
{
	theta = PI*angle/180;
	V = (1/cos(theta))*(sqrt(4.9*pow(distance,2)/(-h_end + h + distance*tan(theta))));
	return V;
}
float shooterV4(float h, float distance, float angle)
{
	theta = PI*angle/180;
	V = (m*g*distance)/(b*h - ((pow(m,2))*g*(exp(-distance)-1)));
	return V;
}
float shooterV5(float distance, float theta)
{
	h = 0.002357*(pow(distance,3)) + 0.001569*(pow(distance,2)) + (0.00487*distance) - (9.91*(pow(10,-5))) + 0.6;
	V = sqrt(((pow(distance,2)*g)/(8*H*(pow(cos(theta),2)))));
	return V;
}
