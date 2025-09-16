<div align="center" style="margin-bottom: 2rem;">
  <h1 style="background: linear-gradient(135deg, #DDA0DD 0%, #E6E6FA 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3.5rem; font-family: 'Georgia', serif; margin-bottom: 0;">
    Velvet Python
  </h1>
  <p style="color: #8B7D8B; font-size: 1.3rem; font-style: italic; margin-top: 0.5rem;">
    Building Python Mastery Through Working Code
  </p>
  <p style="color: #706B70; font-size: 1rem;">
    by Cazzy Aporbo, MS
  </p>
</div>

<div style="background: linear-gradient(135deg, #FFE4E1 0%, #F0E6FF 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
  <h2 style="color: #4A4A4A; margin-top: 0;">Welcome to Your Python Journey</h2>
  <p style="color: #706B70; font-size: 1.1rem; line-height: 1.8;">
    I created Velvet Python after years of learning Python the hard way - through broken production code, 
    failed deployments, and countless hours of debugging. This isn't just another tutorial. It's a 
    comprehensive learning system built from real-world experience, designed to take you from writing 
    scripts to architecting production systems.
  </p>
</div>

## What Makes Velvet Python Different

<div class="grid cards" style="margin: 2rem 0;">

<div style="background: linear-gradient(135deg, #FFF0F5 0%, #FFE4E1 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
<h3 style="color: #8B7D8B;">Real Working Code</h3>
<p style="color: #706B70;">
Every concept comes with production-ready code you can actually use. No toy examples or 
abstract concepts. I've tested everything in real projects.
</p>
<a href="modules/index.md" style="color: #DDA0DD; text-decoration: none; font-weight: bold;">View Modules →</a>
</div>

<div style="background: linear-gradient(135deg, #E6E6FA 0%, #F0E6FF 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
<h3 style="color: #8B7D8B;">Performance Measured</h3>
<p style="color: #706B70;">
Every approach is benchmarked. You'll know not just what works, but what works fast. 
I include actual performance numbers from real hardware.
</p>
<a href="resources/benchmarks.md" style="color: #DDA0DD; text-decoration: none; font-weight: bold;">See Benchmarks →</a>
</div>

<div style="background: linear-gradient(135deg, #F0E6FF 0%, #FFF0F5 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
<h3 style="color: #8B7D8B;">Everything Tested</h3>
<p style="color: #706B70;">
100% of the code has tests. You'll learn not just how to write code, but how to verify 
it works. Testing isn't an afterthought here - it's fundamental.
</p>
<a href="modules/18-testing-quality.md" style="color: #DDA0DD; text-decoration: none; font-weight: bold;">Testing Guide →</a>
</div>

<div style="background: linear-gradient(135deg, #FFEFD5 0%, #F5DEB3 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
<h3 style="color: #8B7D8B;">Beautiful by Design</h3>
<p style="color: #706B70;">
Code should be a joy to read. Every module includes stunning visualizations and 
interactive dashboards with our signature pastel aesthetic.
</p>
<a href="getting-started.md#interactive-apps" style="color: #DDA0DD; text-decoration: none; font-weight: bold;">View Examples →</a>
</div>

</div>

## The Learning Path

<div style="background: linear-gradient(135deg, #F0E6FF 0%, #E6E6FA 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0;">

I've organized 23 modules in a carefully designed progression. Each builds on the previous, but they're also standalone if you need to jump to a specific topic.

### Foundation (Modules 1-5)
Start here to build a solid base:

<div style="padding-left: 1.5rem; margin: 1rem 0;">
<p><strong style="color: #8B7D8B;">01 - Environment Management</strong><br/>
<span style="color: #706B70;">Never have "works on my machine" problems again</span></p>

<p><strong style="color: #8B7D8B;">02 - Package Distribution</strong><br/>
<span style="color: #706B70;">Share your code professionally</span></p>

<p><strong style="color: #8B7D8B;">03 - CLI Applications</strong><br/>
<span style="color: #706B70;">Build tools people want to use</span></p>

<p><strong style="color: #8B7D8B;">04 - DateTime Handling</strong><br/>
<span style="color: #706B70;">Handle time correctly (harder than you think!)</span></p>

<p><strong style="color: #8B7D8B;">05 - Text Processing</strong><br/>
<span style="color: #706B70;">Master strings, regex, and encodings</span></p>
</div>

### Applied Skills (Modules 6-14)
Real-world applications:

<div style="padding-left: 1.5rem; margin: 1rem 0;">
<p><strong style="color: #8B7D8B;">06 - NLP Essentials</strong><br/>
<span style="color: #706B70;">Natural language processing that actually works</span></p>

<p><strong style="color: #8B7D8B;">07 - HTTP & APIs</strong><br/>
<span style="color: #706B70;">Consume and build web APIs</span></p>

<p><strong style="color: #8B7D8B;">08 - Database Systems</strong><br/>
<span style="color: #706B70;">From SQLite to PostgreSQL</span></p>

<p><strong style="color: #8B7D8B;">09 - Concurrency Patterns</strong><br/>
<span style="color: #706B70;">Threading, multiprocessing, and async</span></p>

<p><strong style="color: #8B7D8B;">10 - Media Processing</strong><br/>
<span style="color: #706B70;">Images, audio, and video</span></p>
</div>

<a href="modules/index.md" style="color: #DDA0DD; text-decoration: none; font-weight: bold;">See all 23 modules →</a>

</div>

## Quick Start

### 1. Install Velvet Python

<div style="background: #F5F5F5; padding: 1rem; border-radius: 8px; margin: 1rem 0;">

```bash
# Clone the repository
git clone https://github.com/Cazzy-Aporbo/velvet-python.git
cd velvet-python

# Create virtual environment (see Module 01 for why this matters!)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development

# Verify installation
velvet info
```

</div>

### 2. Choose Your Path

<div style="background: linear-gradient(135deg, #FFE4E1 0%, #FFB6C1 100%); padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
<h4 style="color: #4A4A4A;">Complete Beginner</h4>
<p style="color: #706B70;">Start with Module 01 and work through sequentially:</p>
<pre style="background: rgba(255,255,255,0.7); padding: 0.5rem; border-radius: 5px;"><code>velvet start 01-environments
cd 01-environments
streamlit run app.py  # Launch interactive tutorial</code></pre>
</div>

<div style="background: linear-gradient(135deg, #E6E6FA 0%, #D8BFD8 100%); padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
<h4 style="color: #4A4A4A;">Intermediate Developer</h4>
<p style="color: #706B70;">Jump to the topics you need:</p>
<pre style="background: rgba(255,255,255,0.7); padding: 0.5rem; border-radius: 5px;"><code>velvet modules --difficulty Intermediate
velvet start 09-concurrency  # Example: Learn async</code></pre>
</div>

<div style="background: linear-gradient(135deg, #F0E6FF 0%, #DDA0DD 100%); padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
<h4 style="color: #4A4A4A;">Advanced Practitioner</h4>
<p style="color: #706B70;">Explore architecture and optimization:</p>
<pre style="background: rgba(255,255,255,0.7); padding: 0.5rem; border-radius: 5px;"><code>velvet modules --difficulty Advanced
velvet benchmark 19-performance  # Run performance tests</code></pre>
</div>

### 3. Run Interactive Tutorials

Every module includes a Streamlit app with interactive examples:

```bash
velvet run 01-environments --interactive
```

<div style="background: linear-gradient(135deg, #E6E6FA 0%, #DDA0DD 100%); padding: 1.5rem; border-radius: 10px; margin: 2rem 0;">
  <h3 style="color: #4A4A4A; margin-top: 0;">Experience the Pastel Theme</h3>
  <p style="color: #706B70;">
    Our dashboards aren't just functional - they're beautiful. Every visualization uses our 
    carefully crafted pastel color palette. Learning should be visually pleasant!
  </p>
</div>

## My Philosophy

After years of programming, I've learned that:

<div style="background: linear-gradient(135deg, #FFF0F5 0%, #FFEFD5 100%); padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #DDA0DD;">
<p style="color: #4A4A4A; font-weight: bold; margin: 0;">Code is read far more often than it's written</p>
<p style="color: #706B70; margin: 0.5rem 0 0 0;">That's why every example in Velvet Python is written for clarity first, cleverness never.</p>
</div>

<div style="background: linear-gradient(135deg, #FFE4E1 0%, #F0E6FF 100%); padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #DDA0DD;">
<p style="color: #4A4A4A; font-weight: bold; margin: 0;">Perfect is the enemy of done, but done badly is the enemy of maintenance</p>
<p style="color: #706B70; margin: 0.5rem 0 0 0;">We aim for that sweet spot: pragmatic, clean, tested code that you can actually maintain.</p>
</div>

<div style="background: linear-gradient(135deg, #E6E6FA 0%, #F0E6FF 100%); padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #DDA0DD;">
<p style="color: #4A4A4A; font-weight: bold; margin: 0;">If you can't measure it, you can't improve it</p>
<p style="color: #706B70; margin: 0.5rem 0 0 0;">Every optimization is benchmarked. No guessing about performance.</p>
</div>

## What You'll Build

By the end of this journey, you'll have built:

<div style="background: linear-gradient(135deg, #F0E6FF 0%, #FFE4E1 100%); padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
<ul style="color: #706B70; line-height: 2;">
<li>A complete web application with authentication</li>
<li>Async API clients and servers</li>
<li>Data processing pipelines</li>
<li>Machine learning models in production</li>
<li>Desktop applications with modern UIs</li>
<li>CLI tools that rival commercial products</li>
<li>And much more...</li>
</ul>
</div>

## Join the Community

<div style="display: flex; gap: 1rem; flex-wrap: wrap; margin: 2rem 0;">

<div style="background: linear-gradient(135deg, #FFE4E1 0%, #F0E6FF 100%); padding: 1.5rem; border-radius: 10px; flex: 1; min-width: 250px;">
<h4 style="color: #8B7D8B;">GitHub</h4>
<p style="color: #706B70;">Star the repo, report issues, contribute code</p>
<a href="https://github.com/Cazzy-Aporbo/velvet-python" style="color: #DDA0DD; text-decoration: none; font-weight: bold;">Visit Repository →</a>
</div>

<div style="background: linear-gradient(135deg, #E6E6FA 0%, #FFF0F5 100%); padding: 1.5rem; border-radius: 10px; flex: 1; min-width: 250px;">
<h4 style="color: #8B7D8B;">Discussions</h4>
<p style="color: #706B70;">Ask questions, share projects, help others</p>
<a href="https://github.com/Cazzy-Aporbo/velvet-python/discussions" style="color: #DDA0DD; text-decoration: none; font-weight: bold;">Join Discussions →</a>
</div>

<div style="background: linear-gradient(135deg, #F0E6FF 0%, #FFEFD5 100%); padding: 1.5rem; border-radius: 10px; flex: 1; min-width: 250px;">
<h4 style="color: #8B7D8B;">Blog</h4>
<p style="color: #706B70;">Deep dives into Python concepts</p>
<a href="#" style="color: #DDA0DD; text-decoration: none; font-weight: bold;">Read Articles →</a>
</div>

</div>

## Project Stats

<div style="display: flex; gap: 1rem; flex-wrap: wrap; margin: 2rem 0;">
  <span style="background: linear-gradient(135deg, #DDA0DD 0%, #E6E6FA 100%); color: white; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold;">23 Modules</span>
  <span style="background: linear-gradient(135deg, #E6E6FA 0%, #F0E6FF 100%); color: #4A4A4A; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold;">200+ Examples</span>
  <span style="background: linear-gradient(135deg, #98FB98 0%, #90EE90 100%); color: white; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold;">500+ Tests</span>
  <span style="background: linear-gradient(135deg, #90EE90 0%, #98FB98 100%); color: white; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold;">95% Coverage</span>
</div>

## Acknowledgments

<div style="background: linear-gradient(135deg, #FFF0F5 0%, #FFE4E1 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0;">
<p style="color: #706B70;">
This project exists because of the Python community's generosity in sharing knowledge. 
Special thanks to everyone who's written a tutorial, answered a Stack Overflow question, 
or contributed to open source. This is my way of giving back.
</p>
</div>

## License

<div style="background: linear-gradient(135deg, #F0E6FF 0%, #E6E6FA 100%); padding: 1.5rem; border-radius: 10px; margin: 2rem 0;">
<p style="color: #4A4A4A; margin: 0;"><strong>Code:</strong> MIT License - Use it however you want!</p>
<p style="color: #4A4A4A; margin: 0.5rem 0;"><strong>Documentation:</strong> CC-BY-4.0 - Share and adapt with attribution</p>
<p style="color: #4A4A4A; margin: 0.5rem 0 0 0;"><strong>Author:</strong> Cazzy Aporbo, MS</p>
</div>

---

<div align="center" style="margin-top: 4rem; padding: 2rem; background: linear-gradient(135deg, #FFF0F5 0%, #FFEFD5 100%); border-radius: 15px;">
  <h3 style="color: #8B7D8B; margin-bottom: 1rem;">Ready to Master Python?</h3>
  <p style="color: #706B70; margin-bottom: 1.5rem;">
    Start your journey with Module 01 and build your expertise step by step.
  </p>
  <a href="getting-started/" style="background: linear-gradient(135deg, #DDA0DD 0%, #E6E6FA 100%); color: white; padding: 0.75rem 2rem; border-radius: 25px; text-decoration: none; font-weight: bold; display: inline-block; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.2s;">
    Get Started →
  </a>
</div>

<div style="background: linear-gradient(135deg, #E6E6FA 0%, #F0E6FF 100%); padding: 2rem; border-radius: 15px; margin-top: 2rem;">
<h3 style="color: #8B7D8B;">About the Author</h3>
<p style="color: #706B70;">
<strong>Cazzy Aporbo, MS</strong> is a passionate Python developer who believes in learning by building. 
After years of learning from the community, this is their contribution back - a complete 
learning system that teaches Python the way they wish they had learned it.
</p>
</div>


