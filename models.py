"""
Name: models.py
Author(s): Jared Azevedo & Andres Suarez
Desc: various language models and similarity measurements for matching legal texts with constitutional amendments
"""
import sys
import pandas as pd

from sentence_transformers import SentenceTransformer, util, InputExample, losses
from torch.utils.data import DataLoader


AMENDMENTS = {'First Amendment': 'Congress shall make no law respecting an establishment of religion, or prohibiting the free exercise thereof; or abridging the freedom of speech, or of the press; or the right of the people peaceably to assemble, and to petition the Government for a redress of grievances.',
              'Second Amendment':'A well regulated Militia, being necessary to the security of a free State, the right of the people to keep and bear Arms, shall not be infringed.A well regulated Militia, being necessary to the security of a free State, the right of the people to keep and bear Arms, shall not be infringed.',
              'Third Amendment': 'No Soldier shall, in time of peace be quartered in any house, without the consent of the Owner, nor in time of war, but in a manner to be prescribed by law.',
              'Fourth Amendment':'The right of the people to be secure in their persons, houses, papers, and effects, against unreasonable searches and seizures, shall not be violated, and no Warrants shall issue, but upon probable cause, supported by Oath or affirmation, and particularly describing the place to be searched, and the persons or things to be seized.', 
              'Fifth Amendment': 'No person shall be held to answer for a capital, or otherwise infamous crime, unless on a presentment or indictment of a Grand Jury, except in cases arising in the land or naval forces, or in the Militia, when in actual service in time of War or public danger; nor shall any person be subject for the same offence to be twice put in jeopardy of life or limb; nor shall be compelled in any criminal case to be a witness against himself, nor be deprived of life, liberty, or property, without due process of law; nor shall private property be taken for public use, without just compensation.',
              'Sixth Amendment':'In all criminal prosecutions, the accused shall enjoy the right to a speedy and public trial, by an impartial jury of the State and district wherein the crime shall have been committed, which district shall have been previously ascertained by law, and to be informed of the nature and cause of the accusation; to be confronted with the witnesses against him; to have compulsory process for obtaining witnesses in his favor, and to have the Assistance of Counsel for his defence.',
              'Seventh Amendment': 'In Suits at common law, where the value in controversy shall exceed twenty dollars, the right of trial by jury shall be preserved, and no fact tried by a jury, shall be otherwise re-examined in any Court of the United States, than according to the rules of the common law.', 
              'Eighth Amendment': 'Excessive bail shall not be required, nor excessive fines imposed, nor cruel and unusual punishments inflicted.', 
              'Ninth Amendment': 'The enumeration in the Constitution, of certain rights, shall not be construed to deny or disparage others retained by the people.',
              'Tenth Amendment': 'The powers not delegated to the United States by the Constitution, nor prohibited by it to the States, are reserved to the States respectively, or to the people.',
              'Eleventh Amendment': 'The Judicial power of the United States shall not be construed to extend to any suit in law or equity, commenced or prosecuted against one of the United States by Citizens of another State, or by Citizens or Subjects of any Foreign State.',
              'Twelfth Amendment': 'The Electors shall meet in their respective states and vote by ballot for President and Vice-President, one of whom, at least, shall not be an inhabitant of the same state with themselves; they shall name in their ballots the person voted for as President, and in distinct ballots the person voted for as Vice-President, and they shall make distinct lists of all persons voted for as President, and of all persons voted for as Vice-President, and of the number of votes for each, which lists they shall sign and certify, and transmit sealed to the seat of the government of the United States, directed to the President of the Senate;—The President of the Senate shall, in the presence of the Senate and House of Representatives, open all the certificates and the votes shall then be counted;—The person having the greatest Number of votes for President, shall be the President, if such number be a majority of the whole number of Electors appointed; and if no person have such majority, then from the persons having the highest numbers not exceeding three on the list of those voted for as President, the House of Representatives shall choose immediately, by ballot, the President. But in choosing the President, the votes shall be taken by states, the representation from each state having one vote; a quorum for this purpose shall consist of a member or members from two-thirds of the states, and a majority of all the states shall be necessary to a choice. And if the House of Representatives shall not choose a President whenever the right of choice shall devolve upon them, before the fourth day of March next following, then the Vice-President shall act as President, as in the case of the death or other constitutional disability of the President—The person having the greatest number of votes as Vice-President, shall be the Vice-President, if such number be a majority of the whole number of Electors appointed, and if no person have a majority, then from the two highest numbers on the list, the Senate shall choose the Vice-President; a quorum for the purpose shall consist of two-thirds of the whole number of Senators, and a majority of the whole number shall be necessary to a choice. But no person constitutionally ineligible to the office of President shall be eligible to that of Vice-President of the United States.',
              'Thirteenth Amendment': 'Neither slavery nor involuntary servitudeSupreme Court Rulings- Thirteenth Amendment, Abolishing Slavery: Scott v. Sanford (1857) Court denied citizenship to persons of African descent, and also deprived the Federal government the power to free slaves under the due process clause of the Fifth Amendment. This ruling was indirectly overturned by Thirteenth and Fourteenth Amendments., except as a punishment for crime whereof the party shall have been duly convicted, shall exist within the United States, or any place subject to their jurisdiction. §2 Congress shall have power to enforce this article by appropriate legislation. ',
              'Fourteenth Amendment':'All persons born or naturalized in the United States and subject to the jurisdiction thereof, are citizens of the United States and of the State wherein they reside. No State shall make or enforce any lawSupreme Court Rulings- Fourteenth Amendment, Applying Federal Restrictions to the States: Civil Rights Cases of 1883 Declared that the Thirteenth and Fourteenth Amendments, though they abolished slavery and granted citizenship to former slaves, did not grant the federal government the power to regulate private acts of segregation. Gitlow v. New York (1925) Established that the Fourteenth Amendment expanded the scope of First Amendment free speech protections to include restrictions on state authority. Edwards v. South Carolina (1963) The Fourteenth Amendment does not permit a State to make criminal the peaceful expression of unpopular views. – Justice Potter Stewart, regarding First Amendment freedoms of speech, assembly, and petition, as applied to the states by the Fourteenth Amendment. Planned Parenthood v. Casey (1992) Established the undue burden standard to abortion cases under the Fourteenth Amendment. Bush v. Gore (2000) Concluded that the recount of the 2000 Presidential election in the state of Florida could not be conducted in compliance with the requirements of equal protection and due process guaranteed under the Fourteenth Amendment, due to variations in county standards. which shall abridge the privileges or immunities of citizens of the United States; nor shall any State deprive any personSupreme Court Rulings- Fourteenth Amendment, Due Process Clause: Roe v. Wade (1973) Determined that State abortion laws violate the due process clause of the Fourteenth Amendment, which, according to the ruling, protects against state action the right to privacy, which included the right of a woman to terminate her pregnancy. of life, liberty, or property, without due process of law; nor deny to any person within its jurisdiction the equal protection of the lawsSupreme Court Rulings- Fourteenth Amendment: Equal Protection Clause: Plessy v. Ferguson (1896) Established the separate but equal provision for public acts of segregation in the states under the Equal Protection Clause of the Fourteenth Amendment. The provision was overruled in Brown v. Board of Education of Topeka, Kansas (1954). Brown v. Board of Education of Topeka, Kansas (1954) Court concludes that separate educational facilities are inherently unequal and therefore violate the Equal Protection Clause of the Fourteenth Amendment, overturning the separate but equal standard established in Plessy v. Ferguson (1896). Regents of the University of California v. Bakke (1978) Upheld affirmative action initiatives, allowing race to be considered in college admissions, as constitutional and not in violation of the Equal Protection Clause of the Fourteenth Amendment.. §2 Representatives shall be apportioned among the several States according to their respective numbers, counting the whole number of persons in each State, excluding Indians not taxed. But when the right to vote at any election for the choice of electors for President and Vice President of the United States, Representatives in Congress, the Executive and Judicial officers of a State, or the members of the Legislature thereof, is denied to any of the male inhabitants of such State, being twenty-one years of age, and citizens of the United States, or in any way abridged, except for participation in rebellion, or other crime, the basis of representation therein shall be reduced in the proportion which the number of such male citizens shall bear to the whole number of male citizens twenty-one years of age in such State. §3 No person shall be a Senator or Representative in Congress, or elector of President and Vice President, or hold any office, civil or military, under the United States, or under any State, who, having previously taken an oath, as a member of Congress, or as an officer of the United States, or as a member of any State legislature, or as an executive or judicial officer of any State, to support the Constitution of the United States, shall have engaged in insurrection or rebellion against the same, or given aid or comfort to the enemies thereof. But Congress may by a vote of two-thirds of each House, remove such disability. §4 The validity of the public debt of the United States, authorized by law, including debts incurred for payment of pensions and bounties for services in suppressing insurrection or rebellion, shall not be questioned. But neither the United States nor any State shall assume or pay any debt or obligation incurred in aid of insurrection or rebellion against the United States, or any claim for the loss or emancipation of any slave; but all such debts, obligations and claims shall be held illegal and void. §5 The Congress shall have power to enforce, by appropriate legislation, the provisions of this article.',
              'Fifteenth Amendment':'The right of citizens of the United States to vote shall not be denied or abridged by the United States or by any State on account of race, color, or previous condition of servitude. §2 The Congress shall have power to enforce this article by appropriate legislation.',
              'Sixteenth Amendment':'The Congress shall have power to lay and collect taxes on incomes, from whatever source derived, without apportionment among the several States, and without regard to any census or enumeration.',
              'Seventeenth Amendment':'The Senate of the United States shall be composed of two Senators from each State, elected by the people thereof, for six years; and each Senator shall have one vote. The electors in each State shall have the qualifications requisite for electors of the most numerous branch of the State legislatures. When vacancies happen in the representation of any State in the Senate, the executive authority of such State shall issue writs of election to fill such vacancies: Provided, That the legislature of any State may empower the executive thereof to make temporary appointments until the people fill the vacancies by election as the legislature may direct. This amendment shall not be so construed as to affect the election or term of any Senator chosen before it becomes valid as part of the Constitution.',
              'Eighteenth Amendment':'After one year from the ratification of this article the manufacture, sale, or transportation of intoxicating liquors within, the importation thereof into, or the exportation thereof from the United States and all territory subject to the jurisdiction thereof for beverage purposes is hereby prohibited. §2 The Congress and the several States shall have concurrent power to enforce this article by appropriate legislation. §3 This article shall be inoperative unless it shall have been ratified as an amendment to the Constitution by the legislatures of the several States, as provided in the Constitution, within seven years from the date of the submission hereof to the States by the Congress.',
              'Nineteenth Amendment':'The right of citizens of the United States to vote shall not be denied or abridged by the United States or by any State on account of sex. Congress shall have power to enforce this article by appropriate legislation.',
              'Twentieth Amendment':'The terms of the President and Vice President shall end at noon on the 20th day of January, and the terms of Senators and Representatives at noon on the 3d day of January, of the years in which such terms would have ended if this article had not been ratified; and the terms of their successors shall then begin. §2 The Congress shall assemble at least once in every year, and such meeting shall begin at noon on the 3d day of January, unless they shall by law appoint a different day. §3 If, at the time fixed for the beginning of the term of the President, the President elect shall have died, the Vice President elect shall become President. If a President shall not have been chosen before the time fixed for the beginning of his term, or if the President elect shall have failed to qualify, then the Vice President elect shall act as President until a President shall have qualified; and the Congress may by law provide for the case wherein neither a President elect nor a Vice President elect shall have qualified, declaring who shall then act as President, or the manner in which one who is to act shall be selected, and such person shall act accordingly until a President or Vice President shall have qualified. §4 The Congress may by law provide for the case of the death of any of the persons from whom the House of Representatives may choose a President whenever the right of choice shall have devolved upon them, and for the case of the death of any of the persons from whom the Senate may choose a Vice President whenever the right of choice shall have devolved upon them. §5 Sections 1 and 2 shall take effect on the 15th day of October following the ratification of this article. §6 This article shall be inoperative unless it shall have been ratified as an amendment to the Constitution by the legislatures of three-fourths of the several States within seven years from the date of its submission.',
              'Twenty first Amendment':'The eighteenth article of amendment to the Constitution of the United States is hereby repealed. §2 The transportation or importation into any State, Territory, or possession of the United States for delivery or use therein of intoxicating liquors, in violation of the laws thereof, is hereby prohibited. §3 This article shall be inoperative unless it shall have been ratified as an amendment to the Constitution by conventions in the several States, as provided in the Constitution, within seven years from the date of the submission hereof to the States by the Congress.',
              'Twenty second Amendment': '§1 No person shall be elected to the office of the President more than twice, and no person who has held the office of President, or acted as President, for more than two years of a term to which some other person was elected President shall be elected to the office of the President more than once. But this Article shall not apply to any person holding the office of President, when this Article was proposed by the Congress, and shall not prevent any person who may be holding the office of President, or acting as President, during the term within which this Article becomes operative from holding the office of President or acting as President during the remainder of such term. §2 This article shall be inoperative unless it shall have been ratified as an amendment to the Constitution by the legislatures of three-fourths of the several States within seven years from the date of its submission to the States by the Congress.',
              'Twenty third Amendment':'§1 The District constituting the seat of Government of the United States shall appoint in such manner as the Congress may direct: A number of electors of President and Vice President equal to the whole number of Senators and Representatives in Congress to which the District would be entitled if it were a State, but in no event more than the least populous State; they shall be in addition to those appointed by the States, but they shall be considered, for the purposes of the election of President and Vice President, to be electors appointed by a State; and they shall meet in the District and perform such duties as provided by the twelfth article of amendment. §2 The Congress shall have power to enforce this article by appropriate legislation.',
              'Twenty fourth Amendment':'§1 The right of citizens of the United States to vote in any primary or other election for President or Vice President for electors for President or Vice President, or for Senator or Representative in Congress, shall not be denied or abridged by the United States or any State by reason of failure to pay any poll tax or other tax. §2 The Congress shall have power to enforce this article by appropriate legislation.',
              'Twenty fifth Amendment':'§1 In case of the removal of the President from office or of his death or resignation, the Vice President shall become PresidentBrief History- Vice Presidential Succession: Until the first presidential vacancy arose in 1841 with the untimely death of President William Henry Harrison, there was great uncertainty as to whether the vice president would become an acting president or fully president. John Tyler insisted that he fully succeeded Harrison–to the point that mail addressed to Acting President Tyler was ignored by the White House. Uncertainty also surrounded the role of the vice president during presidential incapacity. In 1919, President Woodrow Wilson suffered a stroke that paralyzed the left side of his body and significantly weakened him otherwise; that the remainder of his administration was largely shaped by his wife Edith, who acted on his behalf. The ratification of the 25th Amendment in 1967 permanently codified the Tyler Precedent and established procedures for the temporary removal of a president due to incapacity.. §2 Whenever there is a vacancy in the office of the Vice President, the President shall nominate a Vice President who shall take office upon confirmation by a majority vote of both Houses of Congress. §3 Whenever the President transmits to the President pro tempore of the Senate and the Speaker of the House of Representatives his written declaration that he is unable to discharge the powers and duties of his office, and until he transmits to them a written declaration to the contrary, such powers and duties shall be discharged by the Vice President as Acting President. §4 Whenever the Vice President and a majority of either the principal officers of the executive departments or of such other body as Congress may by law provide, transmit to the President pro tempore of the Senate and the Speaker of the House of Representatives their written declaration that the President is unable to discharge the powers and duties of his office, the Vice President shall immediately assume the powers and duties of the office as Acting President. Thereafter, when the President transmits to the President pro tempore of the Senate and the Speaker of the House of Representatives his written declaration that no inability exists, he shall resume the powers and duties of his office unless the Vice President and a majority of either the principal officers of the executive department or of such other body as Congress may by law provide, transmit within four days to the President pro tempore of the Senate and the Speaker of the House of Representatives their written declaration that the President is unable to discharge the powers and duties of his office. Thereupon Congress shall decide the issue, assembling within forty-eight hours for that purpose if not in session. If the Congress, within twenty-one days after receipt of the latter written declaration, or, if Congress is not in session, within twenty-one days after Congress is required to assemble, determines by two-thirds vote of both Houses that the President is unable to discharge the powers and duties of his office, the Vice President shall continue to discharge the same as Acting President; otherwise, the President shall resume the powers and duties of his office.',
              'Twenty sixth Amendment':'§1 The right of citizens of the United States, who are eighteen years of age or older, to vote shall not be denied or abridged by the United States or by any State on account of age. §2 The Congress shall have power to enforce this article by appropriate legislation.',
              'Twenty seventh Amendment':'No law varying the compensation for the services of the Senators and Representatives shall take effect, until an election of Representatives shall have intervened.',
              'None': 'None'}


def get_data(inputpath):
    """
    Name: get_data
    Desc: gets data from csv file
    """
    data = pd.read_csv(inputpath)
    data["Match"] = data["Match"].apply(lambda x: x.replace('\n', ' '))

    return data


def data_to_input_examples(data):
    """
    Name: data_to_input_examples
    Desc: converts data into InputExamples
    """
    examples = []

    for ex in range(len(data)):
        match = "None"
        label = 0.0

        if data.loc[ex]['Match'] in AMENDMENTS:
            match = str(AMENDMENTS[data.loc[ex]['Match']])
            label = float(data.loc[ex]['Label'])

        examples.append(InputExample(texts=[data.loc[ex]['Input'], match], label=label))

    return examples


def make_model(input, measurement):
    """
    Name: make_model
    Desc: if not using pretrained model, we want to create and train our own for the legal text similarity task
    Parameters:
        input - the training data we want to use; note that it may not be partitioned yet into train/test sets
        measurement - the measurement score we are using which may help us decide which loss to use
    """
    pass


def finetune(input, lmodel):
    """
    Name: finetune
    Desc: if using pretrained model, finetune for our legal text similarity task
    Parameters:
        input - the training data we want to use; note that it may not be partitioned yet into train/test sets
        lmodel - the model we are using
        measurement - the measurement score we are using which may help us decide which loss to use
    """
    train_dataloader = DataLoader(input, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(lmodel)

    lmodel.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)



def compute(inputpath, lmodel, measurement):
    """
    Name: compute
    Desc: select language model and similarity measurement then compute 
    """
    amendment_embeddings = None
    embeddings = None
    similarities = None

    data = get_data(inputpath)
    examples = data_to_input_examples(data)

    if lmodel == "bert":
        model = SentenceTransformer('all-mpnet-base-v2')

        finetune(examples, model)

        embeddings = model.encode(data.loc["Input"].tolist())
        amendment_embeddings = model.encode([x for x in AMENDMENTS.values()])
    else:
        raise ValueError("Unknown language model")

    if measurement == "cosine":
        similarities = util.cos_sim(embeddings, amendment_embeddings)
    else:
        raise ValueError("Unknown similarity measurement")
    
    all_combinations = []
    for i in range(len(similarities) - 1):
        for j in range(i + 1, len(similarities)):
            all_combinations.append([similarities[i][j], i, j])
    
    all_combinations = sorted(all_combinations, key=lambda x: x[0], reverse=True)

    print("Top-5 most similar pairs:")
    for _, i, j in all_combinations[:5]:
        print("{} \t {} \t {:.4f}".format(input[i], input[j], similarities[i][j]))


def main():
    """
    Name: main
    Desc: ensure correct arugments have been passed in and then execute program
    """
    if len(sys.argv) != 4:
        raise Exception("usage: python models.py inputpath.csv [languageModel] [similarityMeasurement]")

    input = sys.argv[1]
    lmodel = sys.argv[2]
    measurement = str(sys.argv[3])

    compute(input, lmodel, measurement)


if __name__ == '__main__':
    main()
