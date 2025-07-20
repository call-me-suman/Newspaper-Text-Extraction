import os
import re
import nltk
import numpy as np
import pandas as pd
from datetime import datetime
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline
import pyttsx3
from gtts import gTTS
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Load spaCy model for named entity recognition
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Installing spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class NewspaperProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize summarization pipeline
        print("Initializing summarization model...")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        print("Model initialized.")
        self.tts_engine = pyttsx3.init()
        
        self.tts_engine.setProperty('rate', 150)  
        self.tts_engine.setProperty('volume', 0.9)  
        
        
        voices = self.tts_engine.getProperty('voices')
        if len(voices) > 1:
            self.tts_engine.setProperty('voice', voices[1].id)  
        
        self.processed_articles = []
    
    def preprocess_text(self, text):
        """Clean and preprocess the text"""

        text = text.lower()
    
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        tokens = text.split()
 
        cleaned_tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        
        return " ".join(cleaned_tokens), tokens
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of the text"""

        scores = self.sentiment_analyzer.polarity_scores(text)
        
 
        blob = TextBlob(text)
        
        result = {
            "vader_compound": scores['compound'],
            "vader_pos": scores['pos'],
            "vader_neg": scores['neg'],
            "vader_neu": scores['neu'],
            "textblob_polarity": blob.sentiment.polarity,
            "textblob_subjectivity": blob.sentiment.subjectivity
        }
        
        # Determine overall sentiment
        if scores['compound'] >= 0.05:
            result["overall_sentiment"] = "Positive"
        elif scores['compound'] <= -0.05:
            result["overall_sentiment"] = "Negative"
        else:
            result["overall_sentiment"] = "Neutral"
            
        return result
    
    def extract_entities(self, text):
        """Extract named entities from the text"""
        doc = nlp(text)
        
        entities = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],  
            "LOC": [],  
            "DATE": [],
            "TIME": [],
            "MONEY": [],
            "EVENT": []
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
            
        
        for key in entities:
            entities[key] = list(set(entities[key]))
            
        return entities
    
    def extract_key_phrases(self, text):
        """Extract key phrases using TF-IDF"""
        sentences = sent_tokenize(text)
        
       
        vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
       
        feature_names = vectorizer.get_feature_names_out()
        
       
        key_phrases = []
        for i in range(len(sentences)):
            feature_index = tfidf_matrix[i,:].nonzero()[1]
            tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
            sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
            
            if sorted_scores:
                top_word_indices = [x[0] for x in sorted_scores[:3]]
                top_words = [feature_names[i] for i in top_word_indices]
                key_phrases.append(" ".join(top_words))
        
        return key_phrases
    
    def generate_summary(self, text):
        """Generate a summary of the text"""
        if len(text.split()) > 40:  
            summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        return text
    
    def generate_report(self, analysis_result):
        """Generate a structured report from the analysis results"""
        article = analysis_result
        
        report = []
        report.append(f"NEWS REPORT: {article['title'] if 'title' in article else 'Untitled'}")
        report.append("=" * 50)
        

        report.append("SUMMARY:")
        report.append(article['summary'])
        report.append("")
        
        # Add sentiment information
        report.append("SENTIMENT ANALYSIS:")
        report.append(f"Overall tone: {article['sentiment']['overall_sentiment']}")
        if article['sentiment']['overall_sentiment'] == "Positive":
            report.append("The article has a generally positive tone.")
        elif article['sentiment']['overall_sentiment'] == "Negative":
            report.append("The article has a generally negative tone.")
        else:
            report.append("The article maintains a mostly neutral tone.")
        report.append("")
        
        # Add key entities
        report.append("KEY ELEMENTS:")
        
        if article['entities']['PERSON']:
            report.append(f"People mentioned: {', '.join(article['entities']['PERSON'][:5])}")
        
        if article['entities']['ORG']:
            report.append(f"Organizations: {', '.join(article['entities']['ORG'][:5])}")
        
        locations = article['entities']['GPE'] + article['entities']['LOC']
        if locations:
            report.append(f"Locations: {', '.join(locations[:5])}")
        
        if article['entities']['DATE'] or article['entities']['TIME']:
            times = article['entities']['DATE'] + article['entities']['TIME']
            report.append(f"Timeframe: {', '.join(times[:3])}")
        
        if article['key_phrases']:
            report.append(f"Key topics: {', '.join(article['key_phrases'][:5])}")
        
        report.append("")
        report.append("END OF REPORT")
        
        return "\n".join(report)
    
    def generate_news_reader_script(self, article):
        """Generate a script formatted for a news reader"""
        # Create a structure for the news broadcast
        script = []
        
        # Opening
        if 'GPE' in article['entities'] and article['entities']['GPE']:
            location = article['entities']['GPE'][0]
            script.append(f"This is breaking news from {location}.")
        else:
            script.append("Breaking news.")
            
        script.append(f"{article['title']}.")
        
        # Get original content sentences for better flow
        original_sentences = sent_tokenize(article['content'])
        
        # First paragraph - introduction (first 2-3 sentences from original)
        if len(original_sentences) >= 2:
            script.append(" ".join(original_sentences[:2]))
        
        # Add a quote if available
        quote_added = False
        for sentence in original_sentences:
            if '"' in sentence or "'" in sentence:
                for person in article['entities']['PERSON']:
                    if person in sentence:
                        script.append(f"{person} stated, {sentence}")
                        quote_added = True
                        break
                if quote_added:
                    break
        
        # Add some details from the middle of the article
        middle_start = min(2, len(original_sentences))
        middle_end = min(middle_start + 2, len(original_sentences))
        script.append(" ".join(original_sentences[middle_start:middle_end]))
        
        # Add information about organizations involved
        if article['entities']['ORG']:
            org = article['entities']['ORG'][0]
            script.append(f"{org} was mentioned prominently in connection with this story.")
        
        # Closing
        script.append("We will bring you more details as they become available.")
        script.append("Back to the studio.")
        
        return " ".join(script)
    
    def text_to_speech(self, text, output_file=None, use_gtts=False):
        """Convert text to speech"""
        if use_gtts:
            # Use Google TTS
            if not output_file:
                output_file = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            
            print(f"Generating speech with Google TTS to {output_file}...")
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_file)
            print(f"Report saved as {output_file}")
            return output_file
        else:
            # Use pyttsx3 (offline)
            print("Generating speech with pyttsx3...")
            if output_file:
                self.tts_engine.save_to_file(text, output_file)
                self.tts_engine.runAndWait()
                print(f"Report saved as {output_file}")
                return output_file
            else:
                print("Speaking article...")
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                print("Finished speaking.")
    
    def process_article(self, title, content):
        """Process a newspaper article"""
        print("Processing article...")
        # Preprocess the text
        cleaned_text, tokens = self.preprocess_text(content)
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment(content)
        
        # Extract entities
        entities = self.extract_entities(content)
        
        # Extract key phrases
        key_phrases = self.extract_key_phrases(content)
        
        # Generate summary
        summary = self.generate_summary(content)
        
        # Store the processed article
        processed_article = {
            "title": title,
            "content": content,
            "cleaned_content": cleaned_text,
            "tokens": tokens,
            "sentiment": sentiment,
            "entities": entities,
            "key_phrases": key_phrases,
            "summary": summary,
            "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.processed_articles.append(processed_article)
        print("Article processing complete.")
        
        return processed_article
    
    def generate_voice_report(self, article_analysis, output_file=None, use_gtts=False):
        """Generate a voice report from the article analysis"""
        print("Generating news reader script...")
        report_text = self.generate_news_reader_script(article_analysis)
        print(f"Script: {report_text}")
        return self.text_to_speech(report_text, output_file, use_gtts)

    def read_article_directly(self, article_analysis, output_file=None, use_gtts=False):
        """Read the original article directly in a news reader style"""
        # Format the article content with pauses
        content = article_analysis['content']
        # Add a brief introduction
        intro = f"Now reading: {article_analysis['title']}."
        full_text = intro + " " + content
        
        return self.text_to_speech(full_text, output_file, use_gtts)


# Test the system with sample data
def test_processor():
    # Sample article for testing
    sample_title = "Local Community Rallies to Support Healthcare Workers"
    sample_content = """
    In an inspiring display of community spirit, residents of Springfield came together yesterday to show their appreciation for healthcare workers at Springfield General Hospital. Over 200 people participated in a car parade, honking horns and displaying thank you signs as they drove past the hospital during shift change.
    
    Dr. Sarah Johnson, chief of staff at Springfield General, was overwhelmed by the display. "After months of battling this pandemic, seeing this kind of support from our community means everything to our staff," she said. The hospital has been operating at near capacity since March.
    
    The event was organized by community activist Michael Rodriguez, who used social media to spread the word. "These healthcare workers are our heroes, and we wanted them to know they're not alone in this fight," Rodriguez explained.
    
    Local businesses also contributed, with Robertson's Bakery donating 150 cupcakes and Metro Coffee providing free coffee to all hospital staff during the event. Mayor Patricia Williams attended the parade and announced a new city initiative to provide additional resources to the hospital.
    
    The Springfield Police Department escorted the parade and reported no incidents. According to their estimates, the event lasted approximately 45 minutes and included over 100 vehicles.
    
    Hospital administrators have reported a boost in staff morale following the event and are hoping to work with Rodriguez to make it a monthly occurrence as the pandemic continues.
    """
    
    
    # Initialize the processor
    processor = NewspaperProcessor()
    
    # Process the sample article
    result = processor.process_article(sample_title, sample_content)
    
    # Generate a text report
    text_report = processor.generate_report(result)
    print("\n=== TEXT REPORT ===")
    print(text_report)
    
    # Generate a news reader voice report
    print("\n=== GENERATING NEWS READER AUDIO ===")
    
    # Choose one of these options:
    
    # Option 1: Use pyttsx3 (offline TTS) - will speak immediately through speakers
    processor.generate_voice_report(result)
    
    # Option 2: Save as MP3 file using pyttsx3
    # processor.generate_voice_report(result, "news_report.mp3")
    
    # Option 3: Use Google TTS for better quality (requires internet)
    # processor.generate_voice_report(result, "news_report_google.mp3", use_gtts=True)
    
    # Option 4: Read the full article directly
    # processor.read_article_directly(result)
    
    # Return the processed results for further inspection
    return result, text_report

if __name__ == "__main__":
    print("Testing Newspaper Processor...")
    result, report = test_processor()
    
    # Display some additional analysis details
    print("\n=== DETAILED SENTIMENT ANALYSIS ===")
    for key, value in result['sentiment'].items():
        print(f"{key}: {value}")
    
    print("\n=== EXTRACTED ENTITIES ===")
    for entity_type, entities in result['entities'].items():
        if entities:
            print(f"{entity_type}: {', '.join(entities)}")