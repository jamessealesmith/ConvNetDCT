function print_results(results)
fprintf('\nFinal Results:\n')
fprintf('Training Time = %.2f | ', results(4));
fprintf('Loss = %.4f | Training Accuracy = %.4f | Testing Accuracy = %.4f\n',...
    results(1), results(2), results(3));
end

