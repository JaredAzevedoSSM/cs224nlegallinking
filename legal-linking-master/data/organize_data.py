import json
import os 
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

AMENDMENTS = {'first': 'Congress shall make no law respecting an establishment of religion, or prohibiting the free exercise thereof; or abridging the freedom of speech, or of the press; or the right of the people peaceably to assemble, and to petition the Government for a redress of grievances.',
              'second':'A well regulated Militia, being necessary to the security of a free State, the right of the people to keep and bear Arms, shall not be infringed.A well regulated Militia, being necessary to the security of a free State, the right of the people to keep and bear Arms, shall not be infringed.',
              'third': 'No Soldier shall, in time of peace be quartered in any house, without the consent of the Owner, nor in time of war, but in a manner to be prescribed by law.',
              'fourth':'The right of the people to be secure in their persons, houses, papers, and effects, against unreasonable searches and seizures, shall not be violated, and no Warrants shall issue, but upon probable cause, supported by Oath or affirmation, and particularly describing the place to be searched, and the persons or things to be seized.', 
              'fifth': 'No person shall be held to answer for a capital, or otherwise infamous crime, unless on a presentment or indictment of a Grand Jury, except in cases arising in the land or naval forces, or in the Militia, when in actual service in time of War or public danger; nor shall any person be subject for the same offence to be twice put in jeopardy of life or limb; nor shall be compelled in any criminal case to be a witness against himself, nor be deprived of life, liberty, or property, without due process of law; nor shall private property be taken for public use, without just compensation.',
              'sixth':'In all criminal prosecutions, the accused shall enjoy the right to a speedy and public trial, by an impartial jury of the State and district wherein the crime shall have been committed, which district shall have been previously ascertained by law, and to be informed of the nature and cause of the accusation; to be confronted with the witnesses against him; to have compulsory process for obtaining witnesses in his favor, and to have the Assistance of Counsel for his defence.',
              'seventh': 'In Suits at common law, where the value in controversy shall exceed twenty dollars, the right of trial by jury shall be preserved, and no fact tried by a jury, shall be otherwise re-examined in any Court of the United States, than according to the rules of the common law.', 
              'eighth': 'Excessive bail shall not be required, nor excessive fines imposed, nor cruel and unusual punishments inflicted.', 
              'ninth': 'The enumeration in the Constitution, of certain rights, shall not be construed to deny or disparage others retained by the people.',
              'tenth': 'The powers not delegated to the United States by the Constitution, nor prohibited by it to the States, are reserved to the States respectively, or to the people.',
              'eleventh': 'The Judicial power of the United States shall not be construed to extend to any suit in law or equity, commenced or prosecuted against one of the United States by Citizens of another State, or by Citizens or Subjects of any Foreign State.',
              'twelfth': 'The Electors shall meet in their respective states and vote by ballot for President and Vice-President, one of whom, at least, shall not be an inhabitant of the same state with themselves; they shall name in their ballots the person voted for as President, and in distinct ballots the person voted for as Vice-President, and they shall make distinct lists of all persons voted for as President, and of all persons voted for as Vice-President, and of the number of votes for each, which lists they shall sign and certify, and transmit sealed to the seat of the government of the United States, directed to the President of the Senate;—The President of the Senate shall, in the presence of the Senate and House of Representatives, open all the certificates and the votes shall then be counted;—The person having the greatest Number of votes for President, shall be the President, if such number be a majority of the whole number of Electors appointed; and if no person have such majority, then from the persons having the highest numbers not exceeding three on the list of those voted for as President, the House of Representatives shall choose immediately, by ballot, the President. But in choosing the President, the votes shall be taken by states, the representation from each state having one vote; a quorum for this purpose shall consist of a member or members from two-thirds of the states, and a majority of all the states shall be necessary to a choice. And if the House of Representatives shall not choose a President whenever the right of choice shall devolve upon them, before the fourth day of March next following, then the Vice-President shall act as President, as in the case of the death or other constitutional disability of the President—The person having the greatest number of votes as Vice-President, shall be the Vice-President, if such number be a majority of the whole number of Electors appointed, and if no person have a majority, then from the two highest numbers on the list, the Senate shall choose the Vice-President; a quorum for the purpose shall consist of two-thirds of the whole number of Senators, and a majority of the whole number shall be necessary to a choice. But no person constitutionally ineligible to the office of President shall be eligible to that of Vice-President of the United States.',
              'thirteenth': 'Neither slavery nor involuntary servitudeSupreme Court Rulings- Thirteenth Amendment, Abolishing Slavery: Scott v. Sanford (1857) Court denied citizenship to persons of African descent, and also deprived the Federal government the power to free slaves under the due process clause of the Fifth Amendment. This ruling was indirectly overturned by Thirteenth and Fourteenth Amendments., except as a punishment for crime whereof the party shall have been duly convicted, shall exist within the United States, or any place subject to their jurisdiction. §2 Congress shall have power to enforce this article by appropriate legislation. ',
              'fourteenth':'All persons born or naturalized in the United States and subject to the jurisdiction thereof, are citizens of the United States and of the State wherein they reside. No State shall make or enforce any lawSupreme Court Rulings- Fourteenth Amendment, Applying Federal Restrictions to the States: Civil Rights Cases of 1883 Declared that the Thirteenth and Fourteenth Amendments, though they abolished slavery and granted citizenship to former slaves, did not grant the federal government the power to regulate private acts of segregation. Gitlow v. New York (1925) Established that the Fourteenth Amendment expanded the scope of First Amendment free speech protections to include restrictions on state authority. Edwards v. South Carolina (1963) The Fourteenth Amendment does not permit a State to make criminal the peaceful expression of unpopular views. – Justice Potter Stewart, regarding First Amendment freedoms of speech, assembly, and petition, as applied to the states by the Fourteenth Amendment. Planned Parenthood v. Casey (1992) Established the undue burden standard to abortion cases under the Fourteenth Amendment. Bush v. Gore (2000) Concluded that the recount of the 2000 Presidential election in the state of Florida could not be conducted in compliance with the requirements of equal protection and due process guaranteed under the Fourteenth Amendment, due to variations in county standards. which shall abridge the privileges or immunities of citizens of the United States; nor shall any State deprive any personSupreme Court Rulings- Fourteenth Amendment, Due Process Clause: Roe v. Wade (1973) Determined that State abortion laws violate the due process clause of the Fourteenth Amendment, which, according to the ruling, protects against state action the right to privacy, which included the right of a woman to terminate her pregnancy. of life, liberty, or property, without due process of law; nor deny to any person within its jurisdiction the equal protection of the lawsSupreme Court Rulings- Fourteenth Amendment: Equal Protection Clause: Plessy v. Ferguson (1896) Established the separate but equal provision for public acts of segregation in the states under the Equal Protection Clause of the Fourteenth Amendment. The provision was overruled in Brown v. Board of Education of Topeka, Kansas (1954). Brown v. Board of Education of Topeka, Kansas (1954) Court concludes that separate educational facilities are inherently unequal and therefore violate the Equal Protection Clause of the Fourteenth Amendment, overturning the separate but equal standard established in Plessy v. Ferguson (1896). Regents of the University of California v. Bakke (1978) Upheld affirmative action initiatives, allowing race to be considered in college admissions, as constitutional and not in violation of the Equal Protection Clause of the Fourteenth Amendment.. §2 Representatives shall be apportioned among the several States according to their respective numbers, counting the whole number of persons in each State, excluding Indians not taxed. But when the right to vote at any election for the choice of electors for President and Vice President of the United States, Representatives in Congress, the Executive and Judicial officers of a State, or the members of the Legislature thereof, is denied to any of the male inhabitants of such State, being twenty-one years of age, and citizens of the United States, or in any way abridged, except for participation in rebellion, or other crime, the basis of representation therein shall be reduced in the proportion which the number of such male citizens shall bear to the whole number of male citizens twenty-one years of age in such State. §3 No person shall be a Senator or Representative in Congress, or elector of President and Vice President, or hold any office, civil or military, under the United States, or under any State, who, having previously taken an oath, as a member of Congress, or as an officer of the United States, or as a member of any State legislature, or as an executive or judicial officer of any State, to support the Constitution of the United States, shall have engaged in insurrection or rebellion against the same, or given aid or comfort to the enemies thereof. But Congress may by a vote of two-thirds of each House, remove such disability. §4 The validity of the public debt of the United States, authorized by law, including debts incurred for payment of pensions and bounties for services in suppressing insurrection or rebellion, shall not be questioned. But neither the United States nor any State shall assume or pay any debt or obligation incurred in aid of insurrection or rebellion against the United States, or any claim for the loss or emancipation of any slave; but all such debts, obligations and claims shall be held illegal and void. §5 The Congress shall have power to enforce, by appropriate legislation, the provisions of this article.',
              'fifteenth':'The right of citizens of the United States to vote shall not be denied or abridged by the United States or by any State on account of race, color, or previous condition of servitude. §2 The Congress shall have power to enforce this article by appropriate legislation.',
              'sixteenth':'The Congress shall have power to lay and collect taxes on incomes, from whatever source derived, without apportionment among the several States, and without regard to any census or enumeration.',
              'seventeenth':'The Senate of the United States shall be composed of two Senators from each State, elected by the people thereof, for six years; and each Senator shall have one vote. The electors in each State shall have the qualifications requisite for electors of the most numerous branch of the State legislatures. When vacancies happen in the representation of any State in the Senate, the executive authority of such State shall issue writs of election to fill such vacancies: Provided, That the legislature of any State may empower the executive thereof to make temporary appointments until the people fill the vacancies by election as the legislature may direct. This amendment shall not be so construed as to affect the election or term of any Senator chosen before it becomes valid as part of the Constitution.',
              'eighteenth':'After one year from the ratification of this article the manufacture, sale, or transportation of intoxicating liquors within, the importation thereof into, or the exportation thereof from the United States and all territory subject to the jurisdiction thereof for beverage purposes is hereby prohibited. §2 The Congress and the several States shall have concurrent power to enforce this article by appropriate legislation. §3 This article shall be inoperative unless it shall have been ratified as an amendment to the Constitution by the legislatures of the several States, as provided in the Constitution, within seven years from the date of the submission hereof to the States by the Congress.',
              'nineteenth':'The right of citizens of the United States to vote shall not be denied or abridged by the United States or by any State on account of sex. Congress shall have power to enforce this article by appropriate legislation.',
              'twentieth':'The terms of the President and Vice President shall end at noon on the 20th day of January, and the terms of Senators and Representatives at noon on the 3d day of January, of the years in which such terms would have ended if this article had not been ratified; and the terms of their successors shall then begin. §2 The Congress shall assemble at least once in every year, and such meeting shall begin at noon on the 3d day of January, unless they shall by law appoint a different day. §3 If, at the time fixed for the beginning of the term of the President, the President elect shall have died, the Vice President elect shall become President. If a President shall not have been chosen before the time fixed for the beginning of his term, or if the President elect shall have failed to qualify, then the Vice President elect shall act as President until a President shall have qualified; and the Congress may by law provide for the case wherein neither a President elect nor a Vice President elect shall have qualified, declaring who shall then act as President, or the manner in which one who is to act shall be selected, and such person shall act accordingly until a President or Vice President shall have qualified. §4 The Congress may by law provide for the case of the death of any of the persons from whom the House of Representatives may choose a President whenever the right of choice shall have devolved upon them, and for the case of the death of any of the persons from whom the Senate may choose a Vice President whenever the right of choice shall have devolved upon them. §5 Sections 1 and 2 shall take effect on the 15th day of October following the ratification of this article. §6 This article shall be inoperative unless it shall have been ratified as an amendment to the Constitution by the legislatures of three-fourths of the several States within seven years from the date of its submission.',
              'twenty-first':'The eighteenth article of amendment to the Constitution of the United States is hereby repealed. §2 The transportation or importation into any State, Territory, or possession of the United States for delivery or use therein of intoxicating liquors, in violation of the laws thereof, is hereby prohibited. §3 This article shall be inoperative unless it shall have been ratified as an amendment to the Constitution by conventions in the several States, as provided in the Constitution, within seven years from the date of the submission hereof to the States by the Congress.',
              'twenty-second': '§1 No person shall be elected to the office of the President more than twice, and no person who has held the office of President, or acted as President, for more than two years of a term to which some other person was elected President shall be elected to the office of the President more than once. But this Article shall not apply to any person holding the office of President, when this Article was proposed by the Congress, and shall not prevent any person who may be holding the office of President, or acting as President, during the term within which this Article becomes operative from holding the office of President or acting as President during the remainder of such term. §2 This article shall be inoperative unless it shall have been ratified as an amendment to the Constitution by the legislatures of three-fourths of the several States within seven years from the date of its submission to the States by the Congress.',
              'twenty-third':'§1 The District constituting the seat of Government of the United States shall appoint in such manner as the Congress may direct: A number of electors of President and Vice President equal to the whole number of Senators and Representatives in Congress to which the District would be entitled if it were a State, but in no event more than the least populous State; they shall be in addition to those appointed by the States, but they shall be considered, for the purposes of the election of President and Vice President, to be electors appointed by a State; and they shall meet in the District and perform such duties as provided by the twelfth article of amendment. §2 The Congress shall have power to enforce this article by appropriate legislation.',
              'twenty-fourth':'§1 The right of citizens of the United States to vote in any primary or other election for President or Vice President for electors for President or Vice President, or for Senator or Representative in Congress, shall not be denied or abridged by the United States or any State by reason of failure to pay any poll tax or other tax. §2 The Congress shall have power to enforce this article by appropriate legislation.',
              'twenty-fifth':'§1 In case of the removal of the President from office or of his death or resignation, the Vice President shall become PresidentBrief History- Vice Presidential Succession: Until the first presidential vacancy arose in 1841 with the untimely death of President William Henry Harrison, there was great uncertainty as to whether the vice president would become an acting president or fully president. John Tyler insisted that he fully succeeded Harrison–to the point that mail addressed to Acting President Tyler was ignored by the White House. Uncertainty also surrounded the role of the vice president during presidential incapacity. In 1919, President Woodrow Wilson suffered a stroke that paralyzed the left side of his body and significantly weakened him otherwise; that the remainder of his administration was largely shaped by his wife Edith, who acted on his behalf. The ratification of the 25th Amendment in 1967 permanently codified the Tyler Precedent and established procedures for the temporary removal of a president due to incapacity.. §2 Whenever there is a vacancy in the office of the Vice President, the President shall nominate a Vice President who shall take office upon confirmation by a majority vote of both Houses of Congress. §3 Whenever the President transmits to the President pro tempore of the Senate and the Speaker of the House of Representatives his written declaration that he is unable to discharge the powers and duties of his office, and until he transmits to them a written declaration to the contrary, such powers and duties shall be discharged by the Vice President as Acting President. §4 Whenever the Vice President and a majority of either the principal officers of the executive departments or of such other body as Congress may by law provide, transmit to the President pro tempore of the Senate and the Speaker of the House of Representatives their written declaration that the President is unable to discharge the powers and duties of his office, the Vice President shall immediately assume the powers and duties of the office as Acting President. Thereafter, when the President transmits to the President pro tempore of the Senate and the Speaker of the House of Representatives his written declaration that no inability exists, he shall resume the powers and duties of his office unless the Vice President and a majority of either the principal officers of the executive department or of such other body as Congress may by law provide, transmit within four days to the President pro tempore of the Senate and the Speaker of the House of Representatives their written declaration that the President is unable to discharge the powers and duties of his office. Thereupon Congress shall decide the issue, assembling within forty-eight hours for that purpose if not in session. If the Congress, within twenty-one days after receipt of the latter written declaration, or, if Congress is not in session, within twenty-one days after Congress is required to assemble, determines by two-thirds vote of both Houses that the President is unable to discharge the powers and duties of his office, the Vice President shall continue to discharge the same as Acting President; otherwise, the President shall resume the powers and duties of his office.',
              'twenty-sixth':'§1 The right of citizens of the United States, who are eighteen years of age or older, to vote shall not be denied or abridged by the United States or by any State on account of age. §2 The Congress shall have power to enforce this article by appropriate legislation.',
              'twenty-seventh':'No law varying the compensation for the services of the Senators and Representatives shall take effect, until an election of Representatives shall have intervened.',
              'pending': 'pending'}


class data():
    def __init__(self, directory: str, train_test: str) -> None:
        self.directory = directory
        self.train_test = train_test
        self.dir_list = None
        self.observations = None
        self.amendments_categories = None
        self.training_data = []
        self.train_dataloader = None
        

    def dir_files(self):
        self.dir_list = [x for x in os.listdir(self.directory) if self.train_test in x] 
        return self.dir_list


    def org_data(self):
        self.observations = []
        for file in self.dir_list:
            f = open(file, 'r')
            file_contents = f.read().splitlines()
            for elem in file_contents:
                self.observations.extend(json.loads(elem))
        return self.observations


    def label(self):
        for sample in self.observations:
            sample['label'] = 0
            if len(sample['matches']) != 0:
                sample['label'] = 1
        return self.observations


    def check_amendments(self):
        self.amendments_categories = []
        for x in self.observations:
            if x['label'] == 1:
                if x['matches'][0][0] not in self.amendments_categories:
                    self.amendments_categories.append(x['matches'][0][0])
        return self.amendments_categories 


    def add_amendments(self):
        for sample in self.observations:
            if sample['label'] == 0:
                sample['amendment'] = 0
            elif sample['label'] == 1:
                this_amendment = str.lower(sample['matches'][0][0])
                if 'firs' in this_amendment:
                    sample['amendment'] = 'first'
                elif 'sec' in this_amendment:
                    sample['amendment'] = 'second'
                elif 'thir' in this_amendment:
                    sample['amendment'] = 'third'
                elif 'fourth' in this_amendment:
                    sample['amendment'] = 'fourth'
                elif 'fifth' in this_amendment:
                    sample['amendment'] = 'fifth'
                elif 'sixth' in this_amendment:
                    sample['amendment'] = 'sixth'
                elif 'seventh' in this_amendment:
                    sample['amendment'] = 'seventh'
                elif 'eighth' in this_amendment:
                    sample['amendment'] = 'eighth'
                elif 'nineth' in this_amendment:
                    sample['amendment'] = 'ninth'
                elif 'tent' in this_amendment:
                    sample['amendment'] = 'tenth'
                elif 'eleven' in this_amendment:
                    sample['amendment'] = 'eleventh'
                elif 'twelve' in this_amendment:
                    sample['amendment'] = 'twelfth'
                elif 'thirteen' in this_amendment:
                    sample['amendment'] = 'thirteen'
                elif 'fourteen' in this_amendment:
                    sample['amendment'] = 'fourteenth'  
                elif 'fifteen' in this_amendment:
                    sample['amendment'] = 'fifteenth'
                elif 'sixteen' in this_amendment:
                    sample['amendment'] = 'sixteenth'
                elif 'seventeen' in this_amendment:
                    sample['amendment'] = 'seventeenth'
                elif 'eighteen' in this_amendment:
                    sample['amendment'] = 'eighteenth'
                elif 'nineteen' in this_amendment:
                    sample['amendment'] = 'nineteenth'
                elif 'twentieth' in this_amendment:
                    sample['amendment'] = 'twentieth'
                elif 'twenty-first' in this_amendment:
                    sample['amendment'] = 'twenty-first'
                else: 
                    sample['amendment'] = 'pending'
        return self.observations


    def match_amendments(self):
        for sample in self.observations:
            sample['amendment_text'] = 'No'
            if sample['label'] == 1:
                sample['amendment_text'] = AMENDMENTS[str.lower(sample['amendment'])]
        return self.observations


    def training_format(self):
        for sample in self.observations:
            self.training_data.append(InputExample(texts=[sample['text'], sample['amendment_text']], label = sample['label'] ))
        self.train_dataloader = DataLoader(self.training_data, shuffle = True, batch_size = 16)
        return self.train_dataloader
    

if __name__ == "__main__":
    path = os.getcwd()
    training = data(path, 'full')
    # print(training.directory, '\n', training.train_test)
    dir_list = training.dir_files()
    # print(dir_list)
    obs = training.org_data()
    print(len(obs), obs[:3])
    print(len(training.observations))
    obs2 = training.label()
    print(training.observations[:3])

    categoriaas = training.check_amendments()
    print(categoriaas)
    categorias = training.add_amendments()
    # print(training.observations[:50])
    
    resultado = training.match_amendments()
    print(resultado[25].keys())
    print(resultado[25], type(resultado[25]['label']))
    data_loader = training.training_format()
    print(1)
    print(dir(type(data_loader)), len(training.train_dataloader))
    print(len(training.observations) )
    print(len(training.training_data))


    # model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    # train_loss = losses.CosineSimilarityLoss(model)
    # model.fit(train_objectives=[(training.train_dataloader, train_loss)], epochs=1, warmup_steps=100)
    
















    # count = 0
    # lista = []
    # for observation in training.observations:
    #     if observation['label'] == 1 and count < 50:
    #         lista.append(observation)
    #         count +=1
       
    # print(lista[:49])





