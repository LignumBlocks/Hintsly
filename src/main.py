import os
import pandas as pd
import json
from settings import BASE_DIR, SRC_DIR, DATA_DIR
from process_and_validate import (verify_hacks_from_text, get_queries_for_validation, validate_financial_hack, _validation_retrieval_generation,
                                  get_deep_analysis, enriched_analysis, get_structured_analysis, get_hack_classifications)
from process_from_db import grow_descriptions

def process_a_single_hack(id, source_text):
    result, prompt = verify_hacks_from_text(source_text)
    with open(os.path.join(BASE_DIR, 'results'), 'a+') as f:
        f.write('**Hacks Verification**\n')
        f.write(json.dumps(result))
    title = result['possible hack title']
    summary = result['brief summary']

    query_results, prompt = get_queries_for_validation(title, summary)
    with open(os.path.join(BASE_DIR, 'results'), 'a+') as f:
        f.write('\n\n**Generates validation queries**\n')
        f.write(json.dumps(query_results))

    results, prompt, metadata = _validation_retrieval_generation(id, title, summary)
    links = [item[0] for item in metadata]
    links = ' '.join(set(links))
    with open(os.path.join(BASE_DIR, 'results'), 'a+') as f:
        f.write('\n\n**Hacks Validation**\n')
        f.write(json.dumps(results))
        f.write('\n'+links)

    result_free, result_premium, prompt_free, prompt_premium = get_deep_analysis(title, summary, source_text)
    new_result_free, new_result_premium = grow_descriptions(id, result_free, result_premium)
    with open(os.path.join(BASE_DIR, 'results'), 'a+') as f:
        f.write('\n\n**FREE Hack Description**\n')
        f.write(new_result_free)
        f.write('\n\n**PREMIUM Hack Description**\n')
        f.write(new_result_premium)
    structured_free, structured_premium, _, _ = get_structured_analysis(new_result_free, new_result_premium)
    with open(os.path.join(BASE_DIR, 'results'), 'a+') as f:
        f.write('\n\n**Structured info from Free Description**\n')
        f.write(json.dumps(structured_free))
        f.write('\n\n**Structured info from Premium Description**\n')
        f.write(json.dumps(structured_premium))
        
    result_complexity, result_categories, _, _ = get_hack_classifications(new_result_free)
    with open(os.path.join(BASE_DIR, 'results'), 'a+') as f:
        f.write('\n\n**Classify Hacks**\n')
        f.write('Complexity:\n')
        f.write(json.dumps(result_complexity))
        f.write('\nFinancial categories:\n')
        f.write(json.dumps(result_categories))
    


if __name__ == "__main__":
    process_a_single_hack('@hermoneymastery_video_7286913008788426027',"What are all the accounts you need to make your financial life easier and set yourself up for success? This is one of the questions I get asked most often, so I figured I would just break it down for you By the way, if you go to my bio and you get the money mastery toolkit, it walks you through all of this But I'm gonna go ahead and break it down in this video I do not recommend that you use one bank account for all your money and for the people who are gonna say What do you mean all my money? I only have ten dollars Well having multiple bank accounts is gonna help you to have more than ten dollars at a time For the people who think that this is overkill or too complicated to have multiple accounts Let me tell you the purpose for why we do this this changed my total spending habits when I started doing this And it's because you don't want to have all your money sitting in one bank account and you're just spending you're paying bills out of that account You're spending money you're going out to eat you're going shopping whatever the case may be All your money is just flying out of this one bank account It's very hard to keep track of everything and very hard to know where your money is going But let's get into it You need one checking account just for bills every time that you're paid you send a portion of your pay to your bill account And your bills all come out of this account. You do not spend out of this account. It is only for bills You also need one checking account specifically for spending So when you do your budget and you give yourself some fun money This is where the fun money goes to your spending account that way you can look at it at a glance and say Oh, I only have 30 dollars left to spend you're not looking at your bank account as a whole and not knowing how much you actually have To spend so you're not gonna overspend when you do it like this You need one high yield savings account for your emergency fund as well a High-yield savings account cruise more interest than a regular savings account does so your money is making you money while it's just sitting there My two favorite high-yield savings accounts are Ally Bank and Wealthfront Ally specifically I really love it allows you to have buckets within your savings accounts and your checking accounts So that you can bucket out different things if you want to and as a bonus at least one retirement account You can have multiple if you get a 401k for your job You should have that or you can do a Roth IRA if you make under a certain amount per year And you can contribute however much you can to your retirement You at least need to be putting something away just even if it's a little bit that's better than nothing When you do your budget, I recommend that you cash stuff your variable expenses like grocery gas date night whatever your variables are because this makes it easier to see where your money is going and If you have fun money as one of your variables You can send that to your spending account if you wanted to have a spending account For your sinking funds you could cash stuff this to keep it simple for yourself or you could put these in a high-yield savings account with buckets This is what I do with my sinking funds my longer-term funds because it is easier for me Banks are not easily accessible where I currently am so I have all my sinking funds in a high-yield savings Sometimes I cash stuff my variable expenses, but if you're a beginner I would cash stuff both of these things meaning that when you're paid and you do your budget you go to the bank you pull out This amount of money however much you need for them and you put it in different envelopes that are labeled for your funds That's gonna help you see where your money is going how you're spending and it's gonna give you a very good control over your money That is where I started my journey. I hope this helps I hope this gave you some idea of what a good account structure is like I said This is covered in the money mastery toolkit, which will walk you through how to do that So go grab that and if you were wondering this pen is the big Jealousity I get this question every single time so I'm just telling you in advance big jealousy. They're my favorite pens. Have a good day Happy budgeting")

    