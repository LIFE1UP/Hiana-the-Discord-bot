#include <iostream>
#include <vector>

int main(int argc, char *argv[]) {
    std::cout << "s for activate the bot\ni for install the bot(needed at first running)\nu for update\nl for teach the bot\ne for eval the bot\n";
    char context; std::string token;
    std::cout << "cmd: "; std::cin >> context;
    
    switch (context) {
        // interface
    case 's':
        system("python bot.py");
        break;
    case 'i':
        system("pause");
        system("pip install torch");
        system("pip install alive_progress");
        system("pip install numpy");
        system("pip install nltk");
        system("pip install pandas");
        break;
        
    case 'u':
        system("pause");
        system("pip install torch --upgrade");
        system("pip install alive_progress --upgrade");
        system("pip install numpy --upgrade");
        system("pip install nltk --upgrade");
        system("pip install pandas --upgrade");
        break;

    case 'l':
        system("pause");
        system("python train.py");
        break;
    case 'e':
        system("python bot_eval.py txbs/textbook.json pars/data.pth");
        break;
        
    default:
        std::cout << "wrong cmd\n"; exit(0);
    }
    
    return 0;
}
