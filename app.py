"""
Lowe's Semantic Search Web App
===============================
A web-based semantic search engine for finding relevant how-to articles.

Built with Streamlit for easy deployment and user interaction.
"""

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Lowe's How-To Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# SAMPLE DOCUMENTS
# ==============================================================================

SAMPLE_DOCUMENTS = [
    {
        'title': 'How to Install an Underground Sprinkler System',
        'url': 'https://www.lowes.com/n/how-to/install-an-underground-sprinkler-system',
        'content': '''
        Installing an underground sprinkler system can save you time and ensure your lawn gets 
        consistent watering. This guide covers planning your system layout, choosing the right 
        sprinkler heads, digging trenches, installing pipes and valves, and connecting to your 
        water supply. You'll learn about zone planning, water pressure requirements, and timer 
        installation. The project typically takes a weekend and requires basic plumbing skills.
        '''
    },
    {
        'title': 'How to Operate a Standard Riding Lawn Mower',
        'url': 'https://www.lowes.com/n/how-to/operate-a-standard-riding-lawn-mower',
        'content': '''
        Operating a riding lawn mower safely and effectively requires understanding the controls 
        and proper technique. This guide covers starting the engine, adjusting cutting height, 
        steering and turning, mowing patterns for best results, and safety precautions. Learn 
        about blade engagement, speed control, parking brake usage, and maintaining straight lines. 
        We also cover tips for mowing on slopes and around obstacles.
        '''
    },
    {
        'title': 'Leaf Blower Maintenance Guide',
        'url': 'https://www.lowes.com/n/how-to/leaf-blower-maintenance',
        'content': '''
        Proper maintenance keeps your leaf blower running efficiently season after season. This 
        guide covers cleaning air filters, checking spark plugs, inspecting fuel lines, and 
        lubricating moving parts. You'll learn how to winterize your blower, troubleshoot common 
        problems, and extend the life of your equipment. Regular maintenance prevents costly 
        repairs and ensures optimal performance when you need it.
        '''
    },
    {
        'title': 'How to Fertilize Your Lawn',
        'url': 'https://www.lowes.com/n/how-to/fertilize-your-lawn',
        'content': '''
        Fertilizing your lawn properly promotes healthy, green grass and prevents weeds. This 
        guide explains choosing the right fertilizer type, understanding N-P-K ratios, timing 
        applications for your grass type, and using a spreader correctly. Learn about slow-release 
        versus quick-release formulas, organic options, and seasonal feeding schedules. Proper 
        fertilization creates thick, lush turf that resists disease and drought.
        '''
    },
    {
        'title': 'How to Edge Your Lawn',
        'url': 'https://www.lowes.com/n/how-to/how-to-edge-lawn',
        'content': '''
        Clean, crisp edges give your lawn a professionally manicured appearance. This guide covers 
        using manual edgers, power edgers, and string trimmers for edging. You'll learn proper 
        technique for borders along sidewalks, driveways, and flower beds. We explain how to 
        create new edges, maintain existing ones, and achieve straight, even lines. Edging is 
        the finishing touch that makes your yard look polished and well-maintained.
        '''
    },
    {
        'title': 'Building a Raised Bed Vertical Garden',
        'url': 'https://www.lowes.com/n/how-to/raised-bed-vertical-garden',
        'content': '''
        Vertical gardens maximize growing space in small yards and create stunning visual displays. 
        This guide covers building raised beds with vertical supports, choosing climbing plants, 
        installing trellises and support structures, and soil preparation. Learn about irrigation 
        systems for vertical gardens, plant spacing, and maximizing sunlight exposure. Perfect 
        for vegetables, flowers, and herbs in limited spaces.
        '''
    },
    {
        'title': 'How to Prepare Your Lawn for Spring',
        'url': 'https://www.lowes.com/n/how-to/prepare-your-lawn-for-spring',
        'content': '''
        Spring lawn preparation sets the foundation for a beautiful yard all season. This guide 
        covers raking away debris, aerating compacted soil, overseeding thin areas, applying 
        pre-emergent herbicides, and the first fertilizer application. You'll learn when to start 
        based on your climate, how to repair winter damage, and establishing a mowing schedule. 
        Early spring care prevents problems and promotes vigorous growth.
        '''
    },
    {
        'title': 'Best Way to Edge a Garden Bed',
        'url': 'https://www.lowes.com/n/how-to/best-way-to-edge-a-garden-bed',
        'content': '''
        Professional-looking garden bed edges prevent grass invasion and define planting areas. 
        This guide compares edging methods including metal strips, plastic borders, stone, and 
        natural trenches. Learn installation techniques, how deep to edge, maintaining curved 
        versus straight lines, and choosing materials for different landscapes. Proper edging 
        reduces maintenance and creates beautiful transitions between lawn and garden.
        '''
    },
    {
        'title': 'How to Repair Bare Spots in Your Lawn',
        'url': 'https://www.lowes.com/n/how-to/repair-bare-spots-in-your-lawn',
        'content': '''
        Bare spots detract from lawn beauty and invite weeds. This guide covers diagnosing causes 
        like heavy traffic, pet damage, or disease, then repairing with seed or sod. Learn soil 
        preparation, choosing matching grass varieties, proper watering schedules, and protecting 
        new growth. We explain quick fixes versus long-term solutions, and preventing future 
        bare spots through better lawn care practices.
        '''
    }
]

# ==============================================================================
# CACHING FOR PERFORMANCE
# ==============================================================================

@st.cache_resource
def load_model():
    """Load the sentence transformer model (cached for performance)."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def create_embeddings(_model):
    """Create embeddings for all documents (cached for performance)."""
    doc_texts = [f"{doc['title']} {doc['content']}" for doc in SAMPLE_DOCUMENTS]
    return _model.encode(doc_texts, show_progress_bar=False)

# ==============================================================================
# SEARCH FUNCTIONALITY
# ==============================================================================

def search(query, documents, doc_embeddings, model, top_k=3):
    """
    Performs semantic search to find most relevant documents.
    
    Args:
        query (str): User's search query
        documents (list): List of document dictionaries
        doc_embeddings (numpy.ndarray): Pre-computed document embeddings
        model: Sentence transformer model
        top_k (int): Number of results to return
        
    Returns:
        list: Top matching documents with similarity scores
    """
    # Convert query to embedding
    query_embedding = model.encode([query])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    # Get top k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Build results
    results = []
    for idx in top_indices:
        results.append({
            'rank': len(results) + 1,
            'title': documents[idx]['title'],
            'url': documents[idx]['url'],
            'similarity_score': float(similarities[idx]),
            'content': documents[idx]['content'].strip()
        })
    
    return results

# ==============================================================================
# UI COMPONENTS
# ==============================================================================

def display_result_card(result, is_top=False):
    """
    Display a single search result as a card.
    
    Args:
        result (dict): Search result dictionary
        is_top (bool): Whether this is the top result
    """
    if is_top:
        # Top result with special styling
        st.markdown(f"""
        <div style="
            border: 3px solid #4CAF50;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 24px; margin-right: 10px;">🏆</span>
                <span style="
                    background: #4CAF50;
                    color: white;
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-weight: bold;
                    font-size: 14px;
                ">TOP MATCH - {result['similarity_score']:.1%} CONFIDENCE</span>
            </div>
            <h2 style="color: #2c3e50; margin: 10px 0;">{result['title']}</h2>
            <a href="{result['url']}" target="_blank" style="
                color: #3498db;
                text-decoration: none;
                font-size: 14px;
            ">🔗 View Article</a>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Other results with simpler styling
        st.markdown(f"""
        <div style="
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            background: #f9f9f9;
        ">
            <div style="color: #7f8c8d; font-size: 12px; margin-bottom: 5px;">
                #{result['rank']} • {result['similarity_score']:.1%} confidence
            </div>
            <h3 style="color: #34495e; margin: 5px 0; font-size: 18px;">{result['title']}</h3>
            <a href="{result['url']}" target="_blank" style="
                color: #3498db;
                text-decoration: none;
                font-size: 13px;
            ">🔗 View Article</a>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    """Main application function."""
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #2c3e50;">🔍 Lowe's How-To Search</h1>
        <p style="color: #7f8c8d; font-size: 18px;">
            AI-powered semantic search for home improvement articles
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This app uses **AI-powered semantic search** to find relevant how-to articles.
        
        Unlike keyword search, it understands the **meaning** of your query!
        """)
        
        st.header("🎯 Try These Queries")
        example_queries = [
            "How do I fix my lawn?",
            "How do I mow my lawn?",
            "How do I grow healthy greener lawn?",
            "How do I take care of my lawn?"
        ]
        
        for query in example_queries:
            if st.button(query, use_container_width=True):
                st.session_state.query = query
        
        st.header("⚙️ Settings")
        num_results = st.slider("Number of results", 1, 5, 3)
        
        st.header("📚 How It Works")
        st.write("""
        1. **Your query** → Converted to AI embedding
        2. **Documents** → Pre-converted to embeddings
        3. **Comparison** → Cosine similarity calculation
        4. **Results** → Ranked by relevance!
        """)
    
    # Load model and embeddings
    with st.spinner("Loading AI model..."):
        model = load_model()
        doc_embeddings = create_embeddings(model)
    
    # Search input
    query = st.text_input(
        "What do you want to learn?",
        value=st.session_state.get('query', ''),
        placeholder="e.g., How do I maintain my lawn?",
        help="Ask a question about home improvement or lawn care"
    )
    
    # Search button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Perform search
    if search_button and query:
        with st.spinner("Searching..."):
            results = search(query, SAMPLE_DOCUMENTS, doc_embeddings, model, top_k=num_results)
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Search Results")
        
        # Top result
        if results:
            display_result_card(results[0], is_top=True)
            
            # Other results
            if len(results) > 1:
                st.markdown("### Other Relevant Matches")
                for result in results[1:]:
                    display_result_card(result, is_top=False)
        
        # Statistics
        st.markdown("---")
        with st.expander("📈 View Search Statistics"):
            st.write(f"**Query:** {query}")
            st.write(f"**Documents Searched:** {len(SAMPLE_DOCUMENTS)}")
            st.write(f"**Results Returned:** {len(results)}")
            st.write(f"**Top Match Confidence:** {results[0]['similarity_score']:.1%}")
    
    elif search_button and not query:
        st.warning("⚠️ Please enter a search query!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #95a5a6; font-size: 14px;">
        <p>Built with ❤️ using Streamlit & Sentence Transformers</p>
        <p>Learning Project: NLP Semantic Search</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# RUN APP
# ==============================================================================

if __name__ == "__main__":
    main()
