bestData = [0.212,	0.182; 0.557,	0.346; 0.671,	0.671; 0.604,	0.588];

figure(1);
hold off;
bar(bestData);
set(gca,'xticklabel', {"W2V Bayes"; "TFIDF Bayes"; "W2V Tree"; "TFIDF Tree"});
hold on;
plot([2.5, 2.5], [0, 1], "--");
axis([0, 5, 0, 1])
legend("Precision", "Recall");
title("Precision and Recall of the vectorizer and classifier combinations");

depthData = [0.321, 0.337; 0.398, 0.381; 0.537, 0.504; 0.604, 0.588; 0.641, 0.636; 0.664, 0.663];
depthX = [10, 20, 40, 60, 80, 100];

figure(2);
hold off;
scatter(depthX, depthData(:, 1));
hold on;
scatter(depthX, depthData(:, 2));
axis([0,100, 0, 1])
xlabel("Decision Tree Depth");
legend("Precision", "Recall");
title("Precision and Recall for different Decision Tree Depths (TFIDF input)");

vecSizeData = [0.671, 0.671; 0.669, 0.669; 0.669, 0.669;];
vecSizeX = [50, 100, 150];

figure(3);
hold off;
scatter(vecSizeX, vecSizeData(:, 1));
hold on;
scatter(vecSizeX, vecSizeData(:, 2));
axis([40,160, 0, 1])
xlabel("Word2Vec Vector Size");
legend("Precision", "Recall");
title("Precision and Recall for different Word2Vec Vector Sizes (Decision Tree output)");
