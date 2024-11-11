p=randn([2,100000]);
H=diag([10,5]);
an = 30 * 180 / pi;
R=[cos(an),-sin(an);sin(an),cos(an)];
p2=((R*H*p)+[26;61]);
m=mean(p2')';
s=cov(p2')';
x=[-100:0.1:100;-100:0.1:100];
y=funcG(x);
plot(x(1,:),y(1,:),'r',x(2,:),y(2,:),'b');
n=2;
alpha=1;
beta=2;
k=1;
lambda=alpha^2*(n+k)-n
q=chol((n+lambda)*s)';%sqrt(n+lambda*s)

Sigma0=m;
Sigma1=m+q(:,1);
Sigma2=m+q(:,2);
Sigma3=m-q(:,2);
Sigma4=m-q(:,1);

w0m = lambda/(n+lambda);
w0c = lambda/(n+lambda) + (1-alpha^2+beta);
 
w1m = 1/(2*(n+lambda));
w2m = w1m;
w3m = w1m;
w4m = w1m
w1c = w1m;
w2c = w1m;
w3c = w1m;
w4c = w1m;

m2 = w0m*Sigma0+w1m*Sigma1+w2m*Sigma2+w3m*Sigma3+w4m*Sigma4;
s2= w0c*(Sigma0-m2)*(Sigma0-m2)'+w1c*(Sigma1-m2)*(Sigma1-m2)'+w2c*(Sigma2-m2)*(Sigma2-m2)'+w3c*(Sigma3-m2)*(Sigma3-m2)'+w4c*(Sigma4-m2)*(Sigma4-m2)';

Y0=funcG(Sigma0);
Y1=funcG(Sigma1);
Y2=funcG(Sigma2);
Y3=funcG(Sigma3);
Y4=funcG(Sigma4);
my2 = w0m*Y0+w1m*Y1+w2m*Y2+w3m*Y3+w4m*Y4;
sy2= w0c*(Y0-my2)*(Y0-my2)'+w1c*(Y1-my2)*(Y1-my2)'+w2c*(Y2-my2)*(Y2-my2)'+w3c*(Y3-my2)*(Y3-my2)'+w4c*(Y4-my2)*(Y4-my2)';


py2=funcG(p2);
my2Ideal = mean(py2')';
sy2Ideal=cov(py2')';


m, s
m2, s2
my2, sy2
my2Ideal, sy2Ideal

figure(2);
plot(p2(1,:),p2(2,:),'r.', py2(1,:),py2(2,:),'b.')
