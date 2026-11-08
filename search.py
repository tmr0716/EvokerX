#!/usr/bin/env python3
import os
import sys
import time
import requests
import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

load_dotenv()
console = Console(force_terminal=True)

# Active Gemini models list (primary + fallbacks)
CANDIDATE_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite"
]

def fetch_web_results(query: str, max_results: int = 5) -> list:
    """Fetch live web search results via Tavily API."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        console.print("[red]Error:[/red] TAVILY_API_KEY missing in .env file.")
        sys.exit(1)
        
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic"
    }
    
    try:
        res = requests.post(url, json=payload, timeout=12)
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Network Error (Tavily):[/red] {e}")
        return []
        
    if res.status_code != 200:
        console.print(f"[red]Search API Error ({res.status_code}):[/red] {res.text}")
        return []
        
    return res.json().get("results", [])

def synthesize_answer(query: str, search_results: list) -> str:
    """Pass live web search context to Gemini REST API with model fallbacks."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]Error:[/red] GEMINI_API_KEY missing in .env file.")
        sys.exit(1)
        
    context_str = "\n\n".join([
        f"Source [{idx+1}]: {res['title']}\nURL: {res['url']}\nSnippet: {res['content'][:400]}"
        for idx, res in enumerate(search_results)
    ])
    
    prompt = f"""
    You are a concise Search Engine assistant. Synthesize a direct, clear answer based strictly on the context provided.
    Use citations like [1], [2] matching the sources where appropriate.

    User Query: {query}

    Search Context:
    {context_str}
    """

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }
    
    # Try endpoints sequentially if overloads (503) or missing models (404) occur
    for model in CANDIDATE_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
        try:
            res = requests.post(url, json=payload, timeout=30)
            
            if res.status_code in (503, 404):
                console.print(f"[yellow]Endpoint '{model}' ({res.status_code}). Trying fallback model...[/yellow]")
                time.sleep(1)
                continue
                
            if res.status_code != 200:
                console.print(f"[red]Gemini API Error ({res.status_code}):[/red] {res.text}")
                return "Failed to synthesize answer due to API error."
                
            data = res.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
            
        except requests.exceptions.RequestException:
            continue
            
    return "All Gemini model endpoints failed. Please verify API keys and network connection."

def execute_search(search_query: str, num: int, json_out: bool):
    """Executes the search pipeline and prints output."""
    console.print(f"\n[cyan]1/2 Querying Tavily web search for:[/cyan] '{search_query}'...")
    results = fetch_web_results(search_query, max_results=num)
    
    if not results:
        console.print("[yellow]No search results returned.[/yellow]\n")
        return

    if json_out:
        console.print_json(data=results)
        return

    console.print(f"[cyan]2/2 Synthesizing answer with Gemini...[/cyan]")
    ai_response = synthesize_answer(search_query, results)

    console.print("\n" + "=" * 50)
    console.print(Panel(Markdown(ai_response), title="🔍 AI Answer", border_style="cyan"))
    
    console.print("\n[bold yellow]Sources Used:[/bold yellow]")
    for idx, res in enumerate(results, 1):
        console.print(f"[{idx}] [link={res['url']}]{res['title']}[/link] - [dim]{res['url']}[/dim]")
    console.print("=" * 50 + "\n")

@click.command()
@click.argument("query", nargs=-1, required=False)
@click.option("--num", "-n", default=3, help="Number of search sources to fetch.")
@click.option("--json-out", is_flag=True, help="Print raw search results as JSON.")
def main(query, num, json_out):
    """Custom Lightweight LLM-powered Search Engine 🕵️‍♂️"""
    if query:
        # Single command line search mode
        search_query = " ".join(query)
        execute_search(search_query, num, json_out)
    else:
        # Interactive REPL mode
        console.print(Panel("[bold green]Interactive AI Search Mode[/bold green]\nType your query and press Enter. (Type 'exit' or 'quit' to stop)", border_style="green"))
        while True:
            try:
                user_input = console.input("[bold cyan]Search > [/bold cyan]").strip()
                if user_input.lower() in ["exit", "quit", "q"]:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                if not user_input:
                    continue
                execute_search(user_input, num, json_out)
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Goodbye![/yellow]")
                break

if __name__ == "__main__":
    main()