from agents.workflow import run_single_case


def main():
	test_cases = [
		{
			"label": "ඉහළ ගුණාත්මක පිළිතුර",
			"question": "පණ්ඩුකාභය රජතුමාගේ දෙමව්පියන් කවුද?",
			"answer": "පණ්ඩුකාභයගේ පියා දීඝගාමිණී කුමරු වන අතර මව චිත්‍රා කුමරිය ය.",
			"guide": "පියා සහ මව නිවැරදිව සඳහන් කළහොත් පූර්ණ ලකුණු ලබාදෙන්න.",
		},
		{
			"label": "මධ්‍යම ගුණාත්මක පිළිතුර",
			"question": "අනුරාධපුරය සංවර්ධනය කිරීමට පණ්ඩුකාභය කළ දේවල් මොනවාද?",
			"answer": "ඔහු නගරය වටා දිය අගලක් ඉදි කළා. දොරටු සහ මාර්ග පද්ධතියක්ද තිබුණා.",
			"guide": "නගර සංවර්ධනය සහ ජල සම්බන්ධ කරුණු අනුව ලකුණු දෙන්න.",
		},
		{
			"label": "අඩු ගුණාත්මක පිළිතුර",
			"question": "පණ්ඩුකාභය රජතුමාගේ දෙමව්පියන් කවුද?",
			"answer": "එතුමාගේ පියා කාවන්තිස්ස ය. මව විහාරමහාදේවී ය.",
			"guide": "පියා සහ මව නිවැරදිව සඳහන් කළහොත් පූර්ණ ලකුණු ලබාදෙන්න.",
		},
	]

	for idx, case in enumerate(test_cases, start=1):
		print(f"\n================ අවස්ථාව {idx}: {case['label']} ================")
		result = run_single_case(
			question=case["question"],
			student_answer=case["answer"],
			marking_guide=case["guide"],
		)

		print(f"මූලික ලකුණු: {result['final_score']}/20")
		print(f"විශ්වාසතා-සකස් ලකුණු: {result['confidence_adjusted_score']}/20")
		print(f"ආපසුගත් දත්ත විශ්වාසතා අගය: {result['retrieval_confidence']}")
		print("--- ශ්‍රේණිගත කිරීමේ විස්තරය ---")
		print(result["justification"])


if __name__ == "__main__":
	main()
